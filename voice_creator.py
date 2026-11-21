"""Voice creation: a small local LLM plans a web audio search and picks a source; one reference clip results.

Offline step only. Runs in a server worker thread, never from audio callbacks or the realtime path.
"""
from __future__ import annotations

import gc
import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Iterable

from web_audio import ExtractReferenceClip, FetchCandidate, SearchAudio

LOG = logging.getLogger("voicegen")

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LLM_PARAMS = 0.8e9
LLM_THREADS = 4
MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS_LIMIT = 1024
PROMPT_MIN_CHARS = 3
PROMPT_MAX_CHARS = 300
FEEDBACK_MAX_CHARS = 300
QUERY_COUNT = 3
QUERY_MIN_CHARS = 3
QUERY_MAX_CHARS = 120
QUERY_MAX_WORDS = 8
KEYWORD_LIMIT = 6
SEARCH_LIMIT = 10
RANK_POOL = 12
MAX_CANDIDATES_TRIED = 8
MAX_REJECTED_IDS = 200
JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")

GUARD_SYSTEM = (
    "You are a strict safety classifier for a voice-design tool. Decide whether the request asks for the voice of a "
    "specific, named real person (celebrity, politician, streamer, athlete, actor, public figure or private "
    "individual) or asks to impersonate or imitate one. Requests that only describe age, gender, accent, tone, pace, "
    "mood or a fictional character archetype are allowed. Reply with JSON only, in the form "
    '{"is_real_person": true or false, "named_person": "the exact name as written in the request, or an empty string"}.'
)
GUARD_EXAMPLES = (
    ("Request: sound like Keanu Reeves", '{"is_real_person": true, "named_person": "Keanu Reeves"}'),
    ("Request: gentle old grandmother with a scottish accent", '{"is_real_person": false, "named_person": ""}'),
    ("Request: a deep voice of Beyonce", '{"is_real_person": true, "named_person": "Beyonce"}'),
    ("Request: cheerful young woman reading the weather", '{"is_real_person": false, "named_person": ""}'),
)
QUERY_SYSTEM = (
    "You write web search queries that find a recording of one person speaking in a voice that matches a "
    "description. Reply with exactly 3 lines, one search query per line. Each query is 2 to 6 plain words about "
    "the speaker or the kind of recording. No numbering, no quotes, no explanations, never the name of a real person."
)
QUERY_EXAMPLES = (
    (
        "Voice description: friendly middle aged woman cooking show host",
        "middle aged woman cooking show\nfriendly female presenter speech\nwarm woman talking recipe",
    ),
    (
        "Voice description: stern old man army sergeant\nAdjustments requested: louder",
        "old man military speech\nstrong elderly male commanding voice\nstern man giving orders",
    ),
    (
        "Voice description: shy teenage girl vlogger",
        "teenage girl vlog talking\nquiet young female speech\nshy girl personal story",
    ),
)
RANK_SYSTEM = (
    "You choose recordings to use as a clean spoken reference for a voice description. Prefer a single clear speaker, "
    "narration or plain speech, no music, and an open licence. Reply with the labels only, best first, separated by "
    "commas, for example: c2, c3, c1"
)
RANK_EXAMPLES = (
    (
        "Voice description: calm young man reading poetry\n"
        "Recordings (label | title | licence | description):\n"
        "c1 | Heavy metal live concert | CC-BY | loud band and crowd\n"
        "c2 | Poems read aloud by a young man | public domain | single calm speaker reading poetry\n"
        "c3 | City street ambience | CC0 | traffic noise",
        "c2, c3, c1",
    ),
)


# ---------------------------------------------------------------- PUBLIC
class SmallLlm:
    """CPU-only chat model of at most 0.8 B parameters. Not thread-safe; one caller at a time."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._model = None
        self._tokenizer = None
        self._previous_threads = torch.get_num_threads()
        torch.set_num_threads(LLM_THREADS)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
            model = model.to(device="cpu", dtype=torch.float32).eval()
            params = sum(p.numel() for p in model.parameters())
            if params > MAX_LLM_PARAMS:
                raise ValueError(f"{model_id} has {params / 1e9:.2f} B parameters; the limit is {MAX_LLM_PARAMS / 1e9:.1f} B")
        except BaseException:
            torch.set_num_threads(self._previous_threads)
            raise
        self._tokenizer = tokenizer
        self._model = model
        LOG.info("small llm %s loaded (%.0f M params, cpu, %d threads)", model_id, params / 1e6, LLM_THREADS)

    def Chat(self, system: str, user: str, max_new_tokens: int = 256, examples: Iterable[tuple[str, str]] = ()) -> str:
        self._RequireLoaded()
        if not isinstance(system, str) or not isinstance(user, str) or not user.strip():
            raise ValueError("system and user must be strings and user must not be empty")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= MAX_NEW_TOKENS_LIMIT:
            raise ValueError(f"max_new_tokens must be an integer between 1 and {MAX_NEW_TOKENS_LIMIT}")
        messages = [{"role": "system", "content": system}]
        for example_user, example_reply in examples:
            messages.append({"role": "user", "content": example_user})
            messages.append({"role": "assistant", "content": example_reply})
        messages.append({"role": "user", "content": user})
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self._tokenizer(text, return_tensors="pt")
        n_in = int(encoded["input_ids"].shape[1])
        if n_in > MAX_INPUT_TOKENS:
            raise ValueError(f"Prompt is {n_in} tokens; the limit is {MAX_INPUT_TOKENS}")
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._tokenizer.eos_token_id
        with self._torch.inference_mode():
            out = self._model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=pad_id,
            )
        return self._tokenizer.decode(out[0, n_in:], skip_special_tokens=True).strip()

    def ChatJson(self, system: str, user: str, schema_keys: Iterable[str], examples: Iterable[tuple[str, str]] = ()) -> dict | None:
        keys = tuple(schema_keys)
        if not keys or not all(isinstance(k, str) and k for k in keys):
            raise ValueError("schema_keys must be a non-empty collection of non-empty strings")
        return _ExtractJsonObject(self.Chat(system, user, examples=examples), keys)

    def Unload(self) -> None:
        if self._model is None:
            return
        self._model = None
        self._tokenizer = None
        self._torch.set_num_threads(self._previous_threads)
        gc.collect()

    # ---------------------------------------------------------------- PRIVATE
    def _RequireLoaded(self) -> None:
        if self._model is None:
            raise RuntimeError("SmallLlm is unloaded")


class VoiceCreator:
    """Turns a voice description into one reference clip accepted by VCEngine.LoadReference."""

    def __init__(self, vc_engine, voices_dir):
        if not callable(getattr(vc_engine, "LoadReference", None)):
            raise TypeError("vc_engine must provide LoadReference(path)")
        if not isinstance(voices_dir, (str, os.PathLike)) or not str(voices_dir).strip():
            raise ValueError("voices_dir must be a non-empty path")
        self._vc = vc_engine
        self._voices_dir = Path(voices_dir)
        self._voices_dir.mkdir(parents=True, exist_ok=True)

    def Run(
        self,
        job_id: str,
        prompt: str,
        feedback: str | None = None,
        rejected_ids: list[str] | None = None,
        on_progress: Callable[[str], None] = lambda msg: None,
    ) -> dict:
        job_id, prompt, feedback, rejected = self._ValidateRun(job_id, prompt, feedback, rejected_ids, on_progress)

        def progress(message: str) -> None:
            LOG.info("[%s] %s", job_id, message)
            on_progress(message)

        progress("[1/5] Checking the request")
        for text in (prompt, feedback):
            reason = _NamedPersonReason(text)
            if reason:
                raise ValueError(reason)
        hints = self._ExtractFeedbackQueries(feedback)

        llm = None
        try:
            progress("Loading the local language model (CPU)")
            try:
                llm = SmallLlm()
            except Exception as exc:
                raise RuntimeError(f"Local language model failed to load: {type(exc).__name__}: {exc}") from exc
            self._ClassifyWithLlm(llm, f"{prompt}. {feedback}" if feedback else prompt, progress)
            queries, keywords = self._PlanQueries(llm, prompt, hints, rejected, progress)
            candidates, origin = self._SearchAll(queries, rejected, progress)
            ranked = self._RankCandidates(llm, prompt, hints, keywords, candidates, progress)
            llm.Unload()
            return self._TryCandidates(job_id, ranked, origin, progress, _RequestedAttributes(prompt, feedback))
        finally:
            if llm is not None:
                llm.Unload()

    # ---------------------------------------------------------------- PRIVATE
    def _ValidateRun(self, job_id, prompt, feedback, rejected_ids, on_progress):
        if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id):
            raise ValueError("job_id must be 12 lowercase hex characters")
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        prompt = " ".join(prompt.split())
        if not PROMPT_MIN_CHARS <= len(prompt) <= PROMPT_MAX_CHARS:
            raise ValueError(f"prompt must be {PROMPT_MIN_CHARS} to {PROMPT_MAX_CHARS} characters")
        if feedback is None:
            feedback = ""
        if not isinstance(feedback, str):
            raise ValueError("feedback must be a string")
        feedback = " ".join(feedback.split())
        if len(feedback) > FEEDBACK_MAX_CHARS:
            raise ValueError(f"feedback must be at most {FEEDBACK_MAX_CHARS} characters")
        if rejected_ids is not None and not isinstance(rejected_ids, (list, tuple, set, frozenset)):
            raise ValueError("rejected_ids must be a list of source ids")
        if not all(isinstance(i, str) and i for i in rejected_ids or ()):
            raise ValueError("rejected_ids must contain non-empty strings")
        rejected = list(dict.fromkeys(rejected_ids or ()))
        if len(rejected) > MAX_REJECTED_IDS:
            raise ValueError(f"rejected_ids is limited to {MAX_REJECTED_IDS} entries")
        if not callable(on_progress):
            raise ValueError("on_progress must be callable")
        return job_id, prompt, feedback, rejected

    def _ExtractFeedbackQueries(self, feedback: str) -> list[str]:
        """Descriptor phrases for feedback such as 'deeper, slower' -> ['deep low', 'slow calm']."""
        phrases: list[str] = []
        for token in _Tokenize(feedback):
            phrase = _FEEDBACK_HINTS.get(token, token)
            if phrase not in phrases:
                phrases.append(phrase)
        return phrases

    def _ClassifyWithLlm(self, llm: SmallLlm, text: str, progress) -> None:
        verdict = _ParseGuardVerdict(llm.Chat(GUARD_SYSTEM, f"Request: {text}", 60, examples=GUARD_EXAMPLES))
        if verdict is None:
            progress("Safety classifier output was unreadable; the pattern check alone applies")
            return
        flagged, name = verdict
        if not flagged:
            return
        if _IsPlausibleName(name, text):
            raise ValueError(
                f"Requests for the voice of a specific real person ('{name}') are not supported. "
                "Describe the voice instead: age, gender, accent, tone, pace."
            )
        progress("Safety classifier flagged the request but named nobody written in it; allowing")

    def _PlanQueries(self, llm: SmallLlm, prompt, hints, rejected, progress) -> tuple[list[str], list[str]]:
        keywords = _MergeKeywords(hints, prompt)
        fallback = _KeywordQueries(keywords)
        progress("[2/5] Writing search queries")
        lines = [f"Voice description: {prompt}"]
        if hints:
            lines.append("Adjustments requested: " + "; ".join(hints))
        if rejected:
            lines.append("Rejected earlier recordings, find something different: " + "; ".join(_IdToText(i) for i in rejected[-5:]))
        raw = llm.Chat(QUERY_SYSTEM, "\n".join(lines), 80, examples=QUERY_EXAMPLES)
        queries = _ParseQueryLines(raw, avoid=[reply for _, reply in QUERY_EXAMPLES])[:QUERY_COUNT]
        if not queries:
            progress(f"Language model returned no usable queries (raw: {_Clip(raw, 120)!r}); using deterministic keyword queries")
            return fallback, keywords
        if len(queries) < QUERY_COUNT:
            for query in fallback:
                if len(queries) < QUERY_COUNT and query.lower() not in (q.lower() for q in queries):
                    queries.append(query)
            progress("Language model returned fewer than 3 usable queries; filled up with keyword queries")
        progress("Queries: " + " | ".join(queries))
        return queries, keywords

    def _SearchAll(self, queries: list[str], rejected: list[str], progress) -> tuple[list[dict], dict[str, str]]:
        progress("[3/5] Searching public audio sources")
        exclude = set(rejected)
        per_query: list[tuple[str, list[dict]]] = []
        errors: list[str] = []
        for query in queries:
            try:
                found = SearchAudio(query, limit=SEARCH_LIMIT, exclude=exclude)
            except Exception as exc:
                errors.append(f"'{query}': {exc}")
                progress(f"Search failed for '{query}': {exc}")
                continue
            valid = [c for c in found if _IsCandidate(c) and c["id"] not in exclude]
            progress(f"'{query}': {len(valid)} candidates")
            per_query.append((query, valid))
        merged: list[dict] = []
        origin: dict[str, str] = {}
        for rank in range(max((len(valid) for _, valid in per_query), default=0)):
            for query, valid in per_query:
                if rank < len(valid) and valid[rank]["id"] not in origin:
                    origin[valid[rank]["id"]] = query
                    merged.append(valid[rank])
        if not merged:
            detail = "; ".join(errors) if errors else "no source returned a usable result"
            raise RuntimeError(f"No audio candidates found for queries {queries}: {detail}")
        progress(f"{len(merged)} unique candidates")
        return merged, origin

    def _RankCandidates(self, llm: SmallLlm, prompt, hints, keywords, candidates: list[dict], progress) -> list[dict]:
        progress(f"[4/5] Ranking {len(candidates)} candidates")
        labels = {f"c{i + 1}": c for i, c in enumerate(candidates[:RANK_POOL])}
        lines = [f"Voice description: {prompt}"]
        if hints:
            lines.append("Adjustments requested: " + "; ".join(hints))
        lines.append("Recordings (label | title | licence | description):")
        lines.extend(
            f"{label} | {_Clip(c.get('title'), 80)} | {_Clip(c.get('license'), 30)} | {_Clip(c.get('description'), 100)}"
            for label, c in labels.items()
        )
        picked = _ParseRanking(llm.Chat(RANK_SYSTEM, "\n".join(lines), 40, examples=RANK_EXAMPLES), labels)
        rest = _RankByOverlap(keywords, [c for c in candidates if c not in picked])
        if not picked:
            progress("Language model ranking was invalid; using keyword-overlap ranking")
        elif rest:
            progress(f"Language model ranked {len(picked)}; the rest are ordered by keyword overlap")
        else:
            progress(f"Language model ranked all {len(picked)}")
        return picked + rest

    def _TryCandidates(self, job_id: str, ranked: list[dict], origin: dict[str, str], progress, wanted=(None, None)) -> dict:
        clip_path = self._voices_dir / f"{job_id}.wav"
        work_dir = tempfile.mkdtemp(prefix="voicegen_")
        reasons: list[str] = []
        held: list[tuple[float, int, dict, dict, float | None, str]] = []
        total = min(len(ranked), MAX_CANDIDATES_TRIED)
        wanted_text = _DescribeWanted(wanted)
        if any(wanted):
            progress(f"Voice-fit check on median F0: {wanted_text}")
        try:
            for index, candidate in enumerate(ranked[:total]):
                label = candidate.get("title") or candidate["id"]
                progress(f"[5/5] Candidate {index + 1}/{total}: {_Clip(label, 80)} ({candidate.get('source', '?')}, {candidate.get('license') or 'licence unknown'})")
                try:
                    stats = self._AcquireClip(candidate, work_dir, index, clip_path)
                except Exception as exc:
                    reasons.append(f"{candidate['id']}: {exc}")
                    progress(f"Skipped: {exc}")
                    continue
                f0 = _MeasureMedianF0(str(clip_path))
                if f0 is None and any(wanted):
                    progress(f"Candidate {index + 1}: median F0 could not be measured; accepting it without the voice-fit check")
                distance, failed = _AttributeMisfit(wanted, f0)
                if failed:
                    progress(f"Candidate {index + 1}: median F0 {f0:.0f} Hz does not fit {' and '.join(failed)}; trying next")
                    kept = os.path.join(work_dir, f"held{index}.wav")
                    shutil.move(str(clip_path), kept)
                    held.append((distance, index, candidate, stats, f0, kept))
                    continue
                result = self._LoadClip(candidate, stats, f0, False, wanted_text, origin, clip_path, reasons, progress)
                if result is not None:
                    return result
            for distance, index, candidate, stats, f0, kept in sorted(held, key=lambda h: (h[0], h[1])):
                progress(
                    f"No candidate fit {wanted_text}; falling back to candidate {index + 1} "
                    f"(median F0 {f0:.0f} Hz, {distance:.0f} Hz combined distance outside the requested ranges, the closest fit): voice-fit check relaxed"
                )
                clip_path.unlink(missing_ok=True)
                shutil.move(kept, str(clip_path))
                result = self._LoadClip(candidate, stats, f0, True, wanted_text, origin, clip_path, reasons, progress)
                if result is not None:
                    return result
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
        raise RuntimeError(f"No usable reference clip from {total} candidates: " + "; ".join(reasons))

    def _LoadClip(self, candidate, stats, f0, relaxed, wanted_text, origin, clip_path: Path, reasons: list[str], progress) -> dict | None:
        """Hand the staged clip to the voice engine; None when the engine rejects it."""
        try:
            info = self._vc.LoadReference(str(clip_path))
        except ValueError as exc:
            clip_path.unlink(missing_ok=True)
            reasons.append(f"{candidate['id']}: rejected by the voice engine: {exc}")
            progress(f"Voice engine rejected the clip: {exc}")
            return None
        except Exception:
            clip_path.unlink(missing_ok=True)
            raise
        progress("Reference voice loaded")
        clip_stats = dict(stats)
        clip_stats["f0_median_hz"] = None if f0 is None else round(f0, 1)
        clip_stats["f0_relaxed"] = relaxed
        clip_stats["requested_attributes"] = wanted_text
        return {
            "clip_path": str(clip_path),
            "source_id": candidate["id"],
            "source_url": candidate.get("page_url") or candidate["url"],
            "source_title": candidate.get("title") or "",
            "license": candidate.get("license") or "",
            "query": origin.get(candidate["id"], ""),
            "reference_info": info,
            "clip_stats": clip_stats,
        }

    def _AcquireClip(self, candidate: dict, work_dir: str, index: int, clip_path: Path) -> dict:
        attempt_dir = os.path.join(work_dir, f"c{index}")
        os.makedirs(attempt_dir)
        try:
            local = FetchCandidate(candidate, attempt_dir)
            staged = os.path.join(attempt_dir, "clip.wav")
            stats = ExtractReferenceClip(local, staged)
            clip_path.unlink(missing_ok=True)
            shutil.move(staged, str(clip_path))
            return stats
        finally:
            shutil.rmtree(attempt_dir, ignore_errors=True)


# ---------------------------------------------------------------- PRIVATE
_NAME = r"[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){0,3}"
_IMPERSONATION_RE = re.compile(r"\bimpersonat\w*", re.IGNORECASE)
_NAMED_PERSON_RES = (
    re.compile(r"\b(?i:voice\s+of)\s+(" + _NAME + r")"),
    re.compile(
        r"\b(?i:sound(?:s|ing)?\s+(?:exactly\s+|just\s+|a\s+lot\s+|very\s+much\s+|much\s+)?like"
        r"|(?:the\s+)?same\s+voice\s+as|(?:imitat|mimic|clon)\w*|copy(?:ing)?)\s+(" + _NAME + r")"
    ),
    re.compile(r"\b(" + _NAME + r")'s\s+(?i:voice)\b"),
)
_DESCRIPTOR_WORDS = frozenset(
    "american british english irish scottish welsh australian canadian southern northern texan french german italian "
    "spanish russian indian african asian european old young god santa grandma grandpa grandmother grandfather mom dad "
    "mother father uncle aunt my your his her their our this that".split()
)
_VOICE_WORDS = frozenset(
    "narrator reader host speaker voice woman women man men girl boy lady guy female male teen teenage child kid childlike "
    "older elder elderly old young middle aged adult calm warm soft deep low high slow fast energetic cheerful tired bright "
    "authoritative gravelly whispery whispering raspy smooth gentle friendly formal casual monotone news newsreader "
    "documentary story storyteller meditation guide podcast audiobook cartoon character robot announcer presenter teacher "
    "professor sergeant anchor radio sports commentator reading narrating speaking talking".split()
)
_BARE_NAME_RE = re.compile(r"\b[A-Z][a-z']+(?:[ ][A-Z][a-z']+){1,3}\b")
_STOPWORDS = frozenset(
    "and are but can for from has have into its let like make more most not onto over please she some than that the "
    "their them then there these they this those very want was were what when which while who whose will with "
    "would you your voice voices sound sounds sounding create give need someone person something".split()
)
_FEEDBACK_HINTS = {
    "deeper": "deep low", "deep": "deep low", "lower": "low deep", "higher": "high", "lighter": "light high",
    "slower": "slow calm", "faster": "fast energetic", "softer": "soft gentle", "quieter": "soft gentle",
    "louder": "strong clear", "stronger": "strong", "older": "elderly", "younger": "young", "warmer": "warm",
    "brighter": "bright", "darker": "dark deep", "calmer": "calm", "clearer": "clear", "smoother": "smooth",
    "rougher": "gravelly", "raspier": "raspy", "friendlier": "friendly warm", "formal": "formal",
    "casual": "casual conversational", "female": "female woman", "male": "male man", "feminine": "female woman",
    "masculine": "male man",
}
_QUERY_SUFFIXES = ("speech", "audiobook narration", "voice recording")
_OPEN_LICENSE_RE = re.compile(r"\b(cc|cc0|creative commons|public ?domain|pd|librivox)\b", re.IGNORECASE)


# Voice-fit check: median voiced F0 ranges in Hz used to test a clip against the requested attributes.
_GENDER_F0_HZ = {"male": (80.0, 165.0), "female": (165.0, 300.0)}
_LOW_MAX_HZ = {"male": 130.0, "female": 190.0, None: 190.0}  # no gender requested: the permissive bound
_HIGH_MIN_HZ = {"male": 200.0, "female": 260.0, None: 200.0}
_MALE_WORDS = frozenset("male man men boy he his him guy".split())
_FEMALE_WORDS = frozenset("female woman women girl she her lady".split())
_LOW_WORDS = frozenset("deep deeper low lower gravelly bass baritone".split())
_HIGH_WORDS = frozenset("high higher bright childlike squeaky".split())
F0_FLOOR_HZ = 60.0
F0_CEIL_HZ = 400.0
F0_MIN_VOICED_FRAMES = 30


def _RequestedAttributes(*texts: str) -> tuple[str | None, str | None]:
    """(gender, pitch) asked for by keyword rules; None means unconstrained, also when the words conflict."""
    words = set(re.findall(r"[a-z]+", " ".join(t or "" for t in texts).lower()))
    male, female = bool(words & _MALE_WORDS), bool(words & _FEMALE_WORDS)
    low, high = bool(words & _LOW_WORDS), bool(words & _HIGH_WORDS)
    gender = "male" if male and not female else "female" if female and not male else None
    pitch = "low" if low and not high else "high" if high and not low else None
    return gender, pitch


def _DescribeWanted(wanted: tuple[str | None, str | None]) -> str:
    gender, pitch = wanted
    parts = [p for p in (gender, f"{pitch} pitch" if pitch else None) if p]
    return ", ".join(parts) or "no constraint"


def _AttributeMisfit(wanted: tuple[str | None, str | None], f0: float | None) -> tuple[float, list[str]]:
    """(Hz outside the requested ranges, the attributes that fail); (0, []) when it fits or F0 is unknown."""
    gender, pitch = wanted
    if f0 is None:
        return 0.0, []
    distance = 0.0
    failed: list[str] = []
    if gender:
        low, high = _GENDER_F0_HZ[gender]
        gap = max(low - f0, f0 - high, 0.0)
        if gap > 0:
            distance += gap
            failed.append(f"{gender} ({low:.0f}-{high:.0f} Hz)")
    if pitch == "low":
        limit = _LOW_MAX_HZ[gender]
        if f0 > limit:
            distance += f0 - limit
            failed.append(f"low pitch (<= {limit:.0f} Hz)")
    elif pitch == "high":
        limit = _HIGH_MIN_HZ[gender]
        if f0 < limit:
            distance += limit - f0
            failed.append(f"high pitch (>= {limit:.0f} Hz)")
    return distance, failed


def _MeasureMedianF0(path: str) -> float | None:
    """Median voiced F0 in Hz (pyworld DIO + StoneMask), or None when it cannot be measured."""
    try:
        import numpy as np
        import pyworld
        import soundfile

        audio, rate = soundfile.read(path, dtype="float64", always_2d=True)
        mono = np.ascontiguousarray(audio.mean(axis=1))
        f0, times = pyworld.dio(mono, rate, f0_floor=F0_FLOOR_HZ, f0_ceil=F0_CEIL_HZ, frame_period=10.0)
        f0 = pyworld.stonemask(mono, f0, times, rate)
    except Exception as exc:
        LOG.warning("median F0 measurement failed for %s: %s: %s", path, type(exc).__name__, exc)
        return None
    voiced = f0[(f0 >= F0_FLOOR_HZ) & (f0 <= F0_CEIL_HZ)]
    return float(np.median(voiced)) if voiced.size >= F0_MIN_VOICED_FRAMES else None


def _ExtractJsonObject(text: str, required_keys: tuple[str, ...] = ()) -> dict | None:
    start = text.find("{")
    while start != -1:
        end = _MatchBrace(text, start)
        if end != -1:
            try:
                value = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                value = None
            if isinstance(value, dict) and all(k in value for k in required_keys):
                return value
        start = text.find("{", start + 1)
    return None


def _MatchBrace(text: str, start: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*\u2022>]+|\(?\d+\s*[.):]|(?i:query)\s*\d*\s*[:.\-])\s*")
_PREAMBLE_RE = re.compile(r"^(?i:here|sure|certainly|these|below|search|queries|okay|ok)\b")
_CAP_SEQUENCE_RE = re.compile(r"\b[A-Z][a-z']+(?:\s+[A-Z][a-z']+)+")


def _QueryHasName(line: str) -> bool:
    """True when a query line names a person: impersonation wording, 'voice of X' forms, or stray capitalised words."""
    if _NamedPersonReason(line):
        return True
    words = line.split()
    if all(w[:1].isupper() for w in words if w[:1].isalpha()):
        return False  # whole line Title Cased, no way to tell names from styling
    if _CAP_SEQUENCE_RE.search(line):
        return True
    return any(w[:1].isupper() and w.lower().strip("'.,") not in _DESCRIPTOR_WORDS for w in words[1:])


def _ParseQueryLines(text, avoid: Iterable[str] = ()) -> list[str]:
    """Plain-text queries, one per line, tolerating numbering, bullets, quotes, a JSON object and a lead-in sentence."""
    if not isinstance(text, str):
        return []
    parsed = _ExtractJsonObject(text, ("queries",))
    if parsed and isinstance(parsed["queries"], list):
        lines = [str(item) for item in parsed["queries"]]
    else:
        lines = [ln for ln in re.split(r"[\r\n]+", text) if ln.strip()]
        if len(lines) == 1 and re.search(r"[,;]", lines[0]):
            lines = re.split(r"[,;]", lines[0])
    banned = {" ".join(re.sub(r"[^\w\s'-]", " ", ln.lower()).split()) for reply in avoid for ln in reply.splitlines()}
    queries: list[str] = []
    for line in lines:
        previous = None
        while previous != line:
            previous = line
            line = _LIST_PREFIX_RE.sub("", line.strip())
        line = line.strip(" \t\"'`*_[]")
        if not line or line.endswith(":") or _PREAMBLE_RE.match(line) or _QueryHasName(line):
            continue
        words = re.sub(r"[^\w\s'-]", " ", line.lower()).split()
        query = " ".join(words)
        if not 2 <= len(words) <= QUERY_MAX_WORDS or len(query) > QUERY_MAX_CHARS or not re.search(r"[a-z]", query):
            continue
        if query in banned or query in queries:
            continue
        queries.append(query)
    return queries


def _ParseRanking(text, labels: dict) -> list[dict]:
    """Candidates in the order the model listed them: 'c2, c1', '2 > 1', a JSON {"ranked": [...]} or a JSON list."""
    if not isinstance(text, str):
        return []
    parsed = _ExtractJsonObject(text, ("ranked",))
    if parsed and isinstance(parsed["ranked"], list):
        text = " ".join(str(item) for item in parsed["ranked"])
    ids = re.findall(r"\bc(\d+)\b", text.lower()) or re.findall(r"\b(\d+)\b", text)
    picked: list[dict] = []
    for number in ids:
        candidate = labels.get(f"c{int(number)}")
        if candidate is not None and candidate not in picked:
            picked.append(candidate)
    return picked


def _ParseGuardVerdict(text) -> tuple[bool, str] | None:
    """(flagged, name) from the classifier output, or None when it says nothing readable."""
    if not isinstance(text, str):
        return None
    parsed = _ExtractJsonObject(text, ("is_real_person",))
    if parsed:
        flag = parsed["is_real_person"]
        flagged = flag.strip().lower() in ("true", "yes") if isinstance(flag, str) else flag is True
        return flagged, str(parsed.get("named_person") or "").strip()
    flag = re.search(r'is_real_person"?\s*[:=]\s*"?(true|false|yes|no)', text, re.IGNORECASE)
    if not flag:
        return None
    name = re.search(r'named_person"?\s*[:=]\s*"([^"]*)', text)
    return flag.group(1).lower() in ("true", "yes"), (name.group(1).strip() if name else "")


def _IsPlausibleName(name: str, text: str) -> bool:
    """A classifier-reported name counts only if it is written in the request as a proper name, not a descriptor."""
    words = name.split()
    if len(name) < 3 or name not in text or not all(w[:1].isupper() for w in words):
        return False
    if len(words) == 1 and text.startswith(name):
        return False  # a lone capitalised first word is just a sentence start
    if len(words) >= 3 and name.strip(" .") == text.strip(" ."):
        return False  # the model echoed a whole Title Cased description back as a "name"
    return not _IsDescription(name)


def _NamedPersonReason(text: str) -> str | None:
    if _IMPERSONATION_RE.search(text):
        return "Impersonation requests are not supported. Describe the voice instead: age, gender, accent, tone, pace."
    for pattern in _NAMED_PERSON_RES:
        for match in pattern.finditer(text):
            name = match.group(1)
            if name.split()[0].lower() not in _DESCRIPTOR_WORDS:
                return _NamedPersonMessage(name)
    # A bare proper name ('Morgan Freeman', 'make me Elon Musk'): 2-4 capitalised words that are not voice vocabulary.
    for match in _BARE_NAME_RE.finditer(text):
        name = match.group()
        if not _IsDescription(name):
            return _NamedPersonMessage(name)
    return None


def _IsDescription(phrase: str) -> bool:
    """More than half the words are voice vocabulary: 'Older Female Narrator' yes, 'Neil Young' no."""
    words = [w.lower().strip("'") for w in phrase.split()]
    return 2 * sum(w in _DESCRIPTOR_WORDS or w in _VOICE_WORDS or w in _STOPWORDS for w in words) > len(words)


def _NamedPersonMessage(name: str) -> str:
    return (
        f"Requests for the voice of a specific real person ('{name}') are not supported. "
        "Describe the voice instead: age, gender, accent, tone, pace."
    )


def _Tokenize(text: str) -> list[str]:
    tokens = (t.strip("'") for t in re.findall(r"[a-z0-9']+", text.lower()))
    return [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]


def _MergeKeywords(hints: list[str], prompt: str) -> list[str]:
    ordered = _Tokenize(" ".join(hints)) + _Tokenize(prompt)
    return list(dict.fromkeys(ordered))[:KEYWORD_LIMIT]


def _KeywordQueries(keywords: list[str]) -> list[str]:
    base = " ".join(keywords[:5]) or "clear"
    return [_CleanQuery(f"{base} {suffix}") or f"{suffix}" for suffix in _QUERY_SUFFIXES]


def _CleanQuery(text) -> str | None:
    if not isinstance(text, str):
        return None
    words = re.sub(r"[^\w\s'-]", " ", text).split()[:QUERY_MAX_WORDS]
    query = " ".join(words)[:QUERY_MAX_CHARS].strip()
    return query if len(query) >= QUERY_MIN_CHARS else None


def _IsCandidate(item) -> bool:
    return isinstance(item, dict) and isinstance(item.get("id"), str) and bool(item["id"]) and isinstance(item.get("url"), str) and bool(item["url"])


def _Clip(value, limit: int) -> str:
    return " ".join(str(value or "").replace("|", " ").split())[:limit]


def _IdToText(source_id: str) -> str:
    return _Clip(re.sub(r"[^A-Za-z0-9]+", " ", source_id.split(":", 1)[-1]), 60)


def _RankByOverlap(keywords: list[str], candidates: list[dict]) -> list[dict]:
    wanted = set(keywords)

    def score(candidate: dict) -> float:
        text = f"{candidate.get('title') or ''} {candidate.get('description') or ''}"
        overlap = len(wanted.intersection(_Tokenize(text)))
        return overlap + (0.5 if _OPEN_LICENSE_RE.search(str(candidate.get("license") or "")) else 0.0)

    return sorted(candidates, key=lambda c: -score(c))
