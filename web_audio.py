"""Web audio search, download and reference-clip extraction for voice creation.

Offline step only: never imported by the realtime audio path. Sources need no API key:
Internet Archive, Wikimedia Commons and DuckDuckGo (results restricted to direct audio URLs).
"""
from __future__ import annotations

import hashlib
import html
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from itertools import zip_longest
from math import gcd
from urllib.parse import quote, urlparse

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

LOG = logging.getLogger("web_audio")

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a")
MAX_DOWNLOAD_BYTES = 60 * 1024 * 1024
HTTP_TIMEOUT_S = 45
API_TIMEOUT_S = 20
DOWNLOAD_DEADLINE_S = 900
MAX_LIMIT = 50
USER_AGENT = "s2s-voicechanger/1.0 (voice creation; reference audio search)"

ARCHIVE_ITEM_ROWS = 6
ARCHIVE_FILES_PER_ITEM = 2
ARCHIVE_AUDIO_FORMATS = ("VBR MP3", "128Kbps MP3", "MP3", "64Kbps MP3", "Ogg Vorbis", "Flac", "WAVE", "Opus")
COMMONS_ENDPOINT = "https://commons.wikimedia.org/w/api.php"
DESCRIPTION_MAX_CHARS = 300

CONTENT_TYPE_EXTENSIONS = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/ogg": ".ogg",
    "application/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".m4a",
}
OPAQUE_CONTENT_TYPES = ("application/octet-stream", "binary/octet-stream")

ANALYSIS_MAX_S = 20 * 60
VAD_SR = 16000
OUT_SR = 24000
MIN_CLIP_S = 6.5
MAX_CLIP_S = 14.0
TARGET_CLIP_S = 10.0
WINDOW_LENGTHS_S = (6.5, 8.0, 10.0, 12.0, 14.0)
FRAME_S = 0.02
GRID_STEP_FRAMES = 10
MIN_SPEECH_RATIO = 0.6
MIN_SNR_DB = 12.0
CLIP_LEVEL = 0.99
MAX_CLIP_FRACTION = 0.001
MIN_NOISE_FRAMES = 5
NOISE_FLOOR_PERCENTILE = 10
STABILITY_WEIGHT = 0.03
DURATION_WEIGHT = 0.01
VAD_MIN_SILENCE_MS = 250
FADE_S = 0.01
FFMPEG_TIMEOUT_S = 180
POWER_EPS = 1e-10

_VAD_LOCK = threading.Lock()
_VAD_MODEL = None


# ---------------------------------------------------------------- PUBLIC
def SearchAudio(query: str, limit: int = 10, exclude: set[str] | None = None) -> list[dict]:
    """Search licence-aware audio sources. Returns dicts {id, title, url, page_url, source, license, description}."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be an integer in 1..{MAX_LIMIT}")
    if exclude is not None and not isinstance(exclude, (set, frozenset)):
        raise ValueError("exclude must be a set of candidate ids")
    excluded = set(exclude or ())
    fetch_limit = min(limit + len(excluded), MAX_LIMIT)
    sources = (("archive.org", _SearchArchive), ("commons", _SearchCommons), ("web", _SearchWeb))

    batches: list[list[dict]] = []
    errors: list[str] = []
    for name, search in sources:
        try:
            batches.append(search(query.strip(), fetch_limit))
        except Exception as exc:
            LOG.warning("audio source %s failed: %s", name, exc)
            errors.append(f"{name}: {exc}")
    if len(errors) == len(sources):
        raise RuntimeError("all audio sources failed: " + "; ".join(errors))

    merged: list[dict] = []
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    for row in zip_longest(*batches):
        for candidate in row:
            if candidate is None:
                continue
            if candidate["id"] in excluded or candidate["id"] in seen_ids or candidate["url"] in seen_urls:
                continue
            seen_ids.add(candidate["id"])
            seen_urls.add(candidate["url"])
            merged.append(candidate)
            if len(merged) == limit:
                return merged
    return merged


def FetchCandidate(candidate: dict, dest_dir: str) -> str:
    """Download a candidate's audio file (60 MB cap). Returns the local path; ValueError on non-audio or oversize."""
    if not isinstance(candidate, dict) or not isinstance(candidate.get("id"), str) or not isinstance(candidate.get("url"), str):
        raise ValueError("candidate must be a dict with string 'id' and 'url'")
    if not isinstance(dest_dir, str) or not dest_dir:
        raise ValueError("dest_dir must be a non-empty string")
    url = candidate["url"]
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme: {url}")

    import requests

    os.makedirs(dest_dir, exist_ok=True)
    deadline = time.monotonic() + DOWNLOAD_DEADLINE_S
    stem = hashlib.sha1(candidate["id"].encode("utf-8")).hexdigest()[:16]
    part_path = os.path.join(dest_dir, stem + ".part")
    try:
        with requests.get(url, headers={"User-Agent": USER_AGENT}, stream=True, timeout=HTTP_TIMEOUT_S) as response:
            response.raise_for_status()
            extension = _ResolveExtension(response.headers.get("Content-Type", ""), url, response.url)
            declared = response.headers.get("Content-Length", "")
            if declared.isdigit() and int(declared) > MAX_DOWNLOAD_BYTES:
                raise ValueError(f"file is {int(declared)} bytes, over the {MAX_DOWNLOAD_BYTES} byte cap")
            received = 0
            with open(part_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > MAX_DOWNLOAD_BYTES:
                        raise ValueError(f"download exceeded the {MAX_DOWNLOAD_BYTES} byte cap")
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"download exceeded {DOWNLOAD_DEADLINE_S} s")
                    handle.write(chunk)
        if received == 0:
            raise ValueError("empty download")
        final_path = os.path.join(dest_dir, stem + extension)
        os.replace(part_path, final_path)
        return final_path
    finally:
        if os.path.exists(part_path):
            os.remove(part_path)


def ExtractReferenceClip(path: str, out_wav: str) -> dict:
    """Pick the best 6.5-14 s speech window, write it as 24 kHz mono PCM16. Raises ValueError('<reason>') if none passes."""
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    if not isinstance(out_wav, str) or not out_wav:
        raise ValueError("out_wav must be a non-empty string")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    mono, sr = _DecodeMono(path)
    if mono.size / sr < MIN_CLIP_S:
        raise ValueError(f"audio is {mono.size / sr:.1f} s, shorter than {MIN_CLIP_S} s")
    x16 = _Resample(mono, sr, VAD_SR)
    x24 = _Resample(mono, sr, OUT_SR)
    segments = _SpeechSegments(x16)
    if not segments:
        raise ValueError("no speech detected")

    start_s, duration_s, stats = _ChooseWindow(x24, segments, OUT_SR)
    first = int(round(start_s * OUT_SR))
    clip = x24[first:first + int(round(duration_s * OUT_SR))].copy()
    fade = int(FADE_S * OUT_SR)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    clip[:fade] *= ramp
    clip[-fade:] *= ramp[::-1]
    out_dir = os.path.dirname(os.path.abspath(out_wav))
    os.makedirs(out_dir, exist_ok=True)
    sf.write(out_wav, np.clip(clip, -1.0, 1.0), OUT_SR, subtype="PCM_16")
    return {"start_s": start_s, "duration_s": duration_s, **stats}


# ---------------------------------------------------------------- PRIVATE
def _Words(text: str, limit: int = 8) -> list[str]:
    return re.findall(r"[^\W_]+", text.lower())[:limit]


def _First(value) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    return value if isinstance(value, str) else ""


def _Plain(value) -> str:
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value)
    if not isinstance(value, str):
        return ""
    text = html.unescape(re.sub(r"<[^>]+>", " ", value))
    return re.sub(r"\s+", " ", text).strip()[:DESCRIPTION_MAX_CHARS]


def _HasAudioExtension(url: str) -> str:
    path = urlparse(url).path.lower()
    for extension in AUDIO_EXTENSIONS:
        if path.endswith(extension):
            return extension
    return ""


def _ResolveExtension(content_type: str, url: str, final_url: str) -> str:
    base = content_type.split(";")[0].strip().lower()
    from_url = _HasAudioExtension(final_url) or _HasAudioExtension(url)
    if base in CONTENT_TYPE_EXTENSIONS:
        return from_url or CONTENT_TYPE_EXTENSIONS[base]
    if base.startswith("audio/"):
        if from_url:
            return from_url
        raise ValueError(f"unsupported audio content-type: {base}")
    if base in OPAQUE_CONTENT_TYPES and from_url:
        return from_url
    raise ValueError(f"not an audio response (content-type {base or 'missing'})")


def _GetJson(url: str, params: dict) -> dict:
    import requests

    response = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=API_TIMEOUT_S)
    response.raise_for_status()
    return response.json()


def _SearchArchive(query: str, limit: int) -> list[dict]:
    words = _Words(query)
    if not words:
        return []
    docs: list[dict] = []
    for joiner in (" AND ", " OR "):
        expression = (
            f"({joiner.join(words)}) AND mediatype:audio AND "
            "(collection:librivox OR licenseurl:*creativecommons.org* OR licenseurl:*publicdomain*)"
        )
        data = _GetJson("https://archive.org/advancedsearch.php", {
            "q": expression,
            "fl[]": ["identifier", "title", "description", "licenseurl", "collection"],
            "sort[]": "downloads desc",
            "rows": min(ARCHIVE_ITEM_ROWS, limit),
            "page": 1,
            "output": "json",
        })
        docs = data["response"]["docs"]
        if docs or len(words) == 1:
            break
    docs = sorted(docs, key=_ArchiveLicenceRank)

    results: list[dict] = []
    failures: list[str] = []
    for doc in docs:
        identifier = doc.get("identifier")
        if not isinstance(identifier, str) or not identifier:
            continue
        licence = _First(doc.get("licenseurl"))
        if not licence:
            collections = doc.get("collection")
            collections = [collections] if isinstance(collections, str) else (collections or [])
            licence = "Public domain (LibriVox)" if "librivox" in collections else ""
        if not licence:
            continue
        try:
            metadata = _GetJson(f"https://archive.org/metadata/{quote(identifier)}", {})
        except Exception as exc:
            LOG.warning("archive.org metadata for %s failed: %s", identifier, exc)
            failures.append(str(exc))
            continue
        title = _Plain(doc.get("title")) or identifier
        description = _Plain(doc.get("description"))
        for name in _PickArchiveFiles(metadata.get("files", []), ARCHIVE_FILES_PER_ITEM):
            results.append({
                "id": f"archive:{identifier}/{name}",
                "title": f"{title} ({name})",
                "url": f"https://archive.org/download/{quote(identifier)}/{quote(name)}",
                "page_url": f"https://archive.org/details/{quote(identifier)}",
                "source": "archive.org",
                "license": licence,
                "description": description,
            })
    if not results and failures:
        raise RuntimeError(f"archive.org metadata failed: {failures[-1]}")
    return results[:limit]


def _ArchiveLicenceRank(doc: dict) -> int:
    licence = _First(doc.get("licenseurl")).lower()
    if not licence or "publicdomain" in licence or "/zero/" in licence:
        return 0
    if "-nc" in licence or "-nd" in licence:
        return 2
    return 1


def _PickArchiveFiles(files: list[dict], count: int) -> list[str]:
    by_format: dict[str, list[str]] = {}
    for entry in files:
        name = entry.get("name", "")
        audio_format = entry.get("format")
        if audio_format not in ARCHIVE_AUDIO_FORMATS or not _HasAudioExtension("/" + name):
            continue
        size = entry.get("size")
        if not (isinstance(size, str) and size.isdigit()) and not isinstance(size, int):
            continue
        if not 0 < int(size) <= MAX_DOWNLOAD_BYTES:
            continue
        by_format.setdefault(audio_format, []).append(name)
    for audio_format in ARCHIVE_AUDIO_FORMATS:
        names = sorted(by_format.get(audio_format, []))
        if names:
            start = len(names) // 2
            return names[start:start + count]
    return []


def _SearchCommons(query: str, limit: int) -> list[dict]:
    words = _Words(query)
    if not words:
        return []
    pages: list[dict] = []
    for joiner in (" ", " OR "):
        data = _GetJson(COMMONS_ENDPOINT, {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": f"({joiner.join(words)}) filetype:audio",
            "gsrnamespace": 6,
            "gsrlimit": min(limit * 2, 20),
            "prop": "imageinfo",
            "iiprop": "url|mime|size|extmetadata",
            "iiextmetadatafilter": "LicenseShortName|UsageTerms|ImageDescription",
        })
        pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda page: page.get("index", 0))
        if pages or len(words) == 1:
            break
    results: list[dict] = []
    for page in pages:
        info = (page.get("imageinfo") or [{}])[0]
        url = info.get("url", "")
        mime = info.get("mime", "")
        if not url or not (mime.startswith("audio/") or mime == "application/ogg") or not _HasAudioExtension(url):
            continue
        if not 0 < info.get("size", 0) <= MAX_DOWNLOAD_BYTES:
            continue
        meta = info.get("extmetadata", {})
        licence = _Plain((meta.get("LicenseShortName") or {}).get("value")) or _Plain((meta.get("UsageTerms") or {}).get("value"))
        if not licence:
            continue
        title = page.get("title", "")
        results.append({
            "id": f"commons:{title}",
            "title": title.removeprefix("File:"),
            "url": url,
            "page_url": info.get("descriptionurl") or url,
            "source": "commons",
            "license": licence,
            "description": _Plain((meta.get("ImageDescription") or {}).get("value")),
        })
        if len(results) == limit:
            break
    return results


def _SearchWeb(query: str, limit: int) -> list[dict]:
    from ddgs import DDGS

    search = DDGS(timeout=API_TIMEOUT_S)
    queries = (f"{query} voice sample audio mp3", f"{query} filetype:mp3 OR filetype:wav OR filetype:ogg")
    hits: list[dict] = []
    failures: list[str] = []
    for text in queries:
        try:
            hits.extend(search.text(text, max_results=limit * 3))
        except Exception as exc:
            LOG.warning("ddgs query failed: %s", exc)
            failures.append(str(exc))
    if len(failures) == len(queries):
        raise RuntimeError(f"ddgs failed: {failures[-1]}")

    results: list[dict] = []
    for hit in hits:
        url = hit.get("href", "")
        if urlparse(url).scheme not in ("http", "https") or not _HasAudioExtension(url):
            continue
        results.append({
            "id": "web:" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:16],
            "title": _Plain(hit.get("title")) or url,
            "url": url,
            "page_url": url,
            "source": "web",
            "license": "unknown",
            "description": _Plain(hit.get("body")),
        })
        if len(results) == limit:
            break
    return results


def _Resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return wav.astype(np.float32, copy=False)
    g = gcd(src_sr, dst_sr)
    return resample_poly(wav, dst_sr // g, src_sr // g).astype(np.float32)


def _DecodeMono(path: str) -> tuple[np.ndarray, int]:
    try:
        with sf.SoundFile(path) as handle:
            sr = handle.samplerate
            data = handle.read(frames=int(ANALYSIS_MAX_S * sr), dtype="float32", always_2d=True)
    except RuntimeError as exc:
        data, sr = _DecodeWithFfmpeg(path, exc)
    if data.size == 0:
        raise ValueError("decoded audio is empty")
    return data.mean(axis=1).astype(np.float32), sr


def _DecodeWithFfmpeg(path: str, cause: Exception) -> tuple[np.ndarray, int]:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise ValueError(f"cannot decode {os.path.basename(path)}: {cause}; ffmpeg is not on PATH")
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "decoded.wav")
        command = [exe, "-v", "error", "-nostdin", "-t", str(ANALYSIS_MAX_S), "-i", path, "-ac", "1", "-ar", str(OUT_SR), "-f", "wav", target]
        try:
            result = subprocess.run(command, capture_output=True, timeout=FFMPEG_TIMEOUT_S, check=False)
        except subprocess.TimeoutExpired as exc:
            raise ValueError("ffmpeg decode timed out") from exc
        if result.returncode != 0 or not os.path.isfile(target):
            detail = result.stderr.decode("utf-8", "replace").strip()[-200:]
            raise ValueError(f"ffmpeg could not decode {os.path.basename(path)}: {detail}")
        data, sr = sf.read(target, dtype="float32", always_2d=True)
    return data, sr


def _VadModel():
    global _VAD_MODEL
    with _VAD_LOCK:
        if _VAD_MODEL is None:
            from silero_vad import load_silero_vad

            _VAD_MODEL = load_silero_vad()
        return _VAD_MODEL


def _SpeechSegments(x16: np.ndarray) -> list[tuple[float, float]]:
    import torch
    from silero_vad import get_speech_timestamps

    with torch.no_grad():
        stamps = get_speech_timestamps(
            torch.from_numpy(np.ascontiguousarray(x16)),
            _VadModel(),
            sampling_rate=VAD_SR,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        )
    return [(stamp["start"] / VAD_SR, stamp["end"] / VAD_SR) for stamp in stamps]


def _ChooseWindow(x: np.ndarray, segments: list[tuple[float, float]], sr: int) -> tuple[float, float, dict]:
    hop = int(round(sr * FRAME_S))
    n_frames = x.size // hop
    frame_s = hop / sr
    if n_frames * frame_s < MIN_CLIP_S:
        raise ValueError(f"audio is {n_frames * frame_s:.1f} s, shorter than {MIN_CLIP_S} s")
    frames = x[:n_frames * hop].reshape(n_frames, hop).astype(np.float64)
    power = np.mean(frames ** 2, axis=1)
    db = 10.0 * np.log10(power + POWER_EPS)
    clipped = np.sum(np.abs(frames) > CLIP_LEVEL, axis=1)

    mask = np.zeros(n_frames, dtype=bool)
    seg_starts: list[int] = []
    seg_ends: list[int] = []
    for start, end in segments:
        first = min(int(start / frame_s), n_frames)
        last = min(int(np.ceil(end / frame_s)), n_frames)
        mask[first:last] = True
        seg_starts.append(first)
        seg_ends.append(last)
    cum_mask = np.concatenate(([0], np.cumsum(mask)))
    cum_clip = np.concatenate(([0], np.cumsum(clipped)))

    best_score = -np.inf
    best: tuple[int, int, float, float, float] | None = None
    counts = {"speech": 0, "clip": 0, "snr": 0}
    best_ratio = 0.0
    best_snr = -np.inf
    for length_s in WINDOW_LENGTHS_S:
        nf = int(round(length_s / frame_s))
        if nf > n_frames:
            continue
        grid = np.arange(0, n_frames - nf + 1, GRID_STEP_FRAMES)
        starts = np.unique(np.clip(np.concatenate((grid, seg_starts, np.asarray(seg_ends) - nf)), 0, n_frames - nf)).astype(int)
        for s in starts:
            e = s + nf
            ratio = (cum_mask[e] - cum_mask[s]) / nf
            best_ratio = max(best_ratio, ratio)
            if ratio < MIN_SPEECH_RATIO:
                counts["speech"] += 1
                continue
            if (cum_clip[e] - cum_clip[s]) / (nf * hop) > MAX_CLIP_FRACTION:
                counts["clip"] += 1
                continue
            window_mask = mask[s:e]
            window_power = power[s:e]
            noise_power = window_power[~window_mask]
            if noise_power.size < MIN_NOISE_FRAMES:
                noise_power = window_power[window_power <= np.percentile(window_power, NOISE_FLOOR_PERCENTILE)]
            snr = 10.0 * np.log10(np.mean(window_power[window_mask]) / (np.mean(noise_power) + POWER_EPS) + POWER_EPS)
            best_snr = max(best_snr, snr)
            if snr < MIN_SNR_DB:
                counts["snr"] += 1
                continue
            stability = float(np.std(db[s:e][window_mask]))
            score = ratio - STABILITY_WEIGHT * stability - DURATION_WEIGHT * abs(nf * frame_s - TARGET_CLIP_S)
            if score > best_score:
                best_score = score
                best = (int(s), int(nf), float(ratio), float(snr), float(10.0 * np.log10(np.mean(window_power) + POWER_EPS)))

    if best is None:
        raise ValueError(
            f"no {MIN_CLIP_S}-{MAX_CLIP_S} s window passed: speech_ratio<{MIN_SPEECH_RATIO} in {counts['speech']} "
            f"(best {best_ratio:.2f}), clipping in {counts['clip']}, SNR<{MIN_SNR_DB:.0f} dB in {counts['snr']} "
            f"(best {f'{best_snr:.1f} dB' if np.isfinite(best_snr) else 'n/a'})"
        )
    s, nf, ratio, snr, rms_db = best
    return round(s * frame_s, 3), round(nf * frame_s, 3), {
        "speech_ratio": round(ratio, 3),
        "rms_db": round(rms_db, 2),
        "snr_db": round(snr, 2),
    }
