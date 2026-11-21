// Seed-VC control page. Built from /api/state at load; every request goes to this origin.
(function () {
  "use strict";

  var TONE_KEYS = ["pitch_shift_st", "formant_shift", "f0_expander", "cfg_rate", "diffusion_steps",
    "left_context_s", "block_ms", "overlap_ms", "timbre_mix", "monitor_volume"];
  var TONE_LABELS = {
    pitch_shift_st: "Pitch shift (semitones)",
    formant_shift: "Formant shift",
    f0_expander: "F0 expressiveness",
    cfg_rate: "CFG rate",
    diffusion_steps: "Diffusion steps",
    left_context_s: "Left context",
    block_ms: "Block size",
    overlap_ms: "SOLA overlap",
    timbre_mix: "Timbre mix (0 dry, 1 wet)",
    monitor_volume: "Monitor volume"
  };
  var UNITS = { pitch_shift_st: " st", f0_expander: " x", left_context_s: " s", block_ms: " ms", overlap_ms: " ms" };
  var BOOL_KEYS = ["monitor_enabled", "auto_balancer"];

  var state = { ranges: {}, defaults: {}, params: {}, slotNames: [], presets: [], recordMax: 10, slots: [], rvc: { models: [], weights: null } };
  var sliders = {};
  var cards = [];
  var recordSlot = null;
  var recordTimer = null;
  var tickTimer = null;
  var recordStarted = 0;
  var WEIGHTS_POLL_MS = 2000;
  var weightsToken = 0;
  var engineText = "";
  var lastEngineError = null;

  function $(id) { return document.getElementById(id); }
  function setText(id, text) { $(id).textContent = text; }

  function say(text, isError) {
    var box = $("msg");
    box.textContent = text;
    box.className = isError ? "msg error" : "msg";
  }

  function fail(err) { say("network error: " + err, true); }

  function fmt(key, value) {
    if (typeof value !== "number" || !isFinite(value)) { return "n/a"; }
    var r = state.ranges[key];
    var digits = r && r.type === "int" ? 0 : 2;
    return value.toFixed(digits) + (UNITS[key] || "");
  }

  function parseResponse(resp) {
    return resp.json().catch(function () { return {}; }).then(function (data) {
      return { ok: resp.ok, status: resp.status, data: data };
    });
  }

  function request(method, url, body) {
    var opts = { method: method, headers: {} };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    return fetch(url, opts).then(parseResponse);
  }

  function errorOf(res) {
    if (res.data && typeof res.data.detail === "string") { return res.data.detail; }
    return "request failed with HTTP " + res.status;
  }

  function button(text, cls) {
    var b = document.createElement("button");
    b.type = "button";
    b.className = cls;
    b.textContent = text;
    return b;
  }

  function setParam(name, value) {
    return request("POST", "/api/param", { name: name, value: value }).then(function (res) {
      if (!res.ok) {
        say(name + ": " + errorOf(res), true);
        syncControls();
        return false;
      }
      state.params[name] = res.data.value;
      say("", false);
      syncControls();
      return true;
    }, function (err) { fail(err); return false; });
  }

  function buildSliders() {
    var host = $("sliders");
    host.textContent = "";
    sliders = {};
    TONE_KEYS.forEach(function (key) {
      var r = state.ranges[key];
      if (!r) { return; }
      var row = document.createElement("div");
      row.className = "control";
      var head = document.createElement("div");
      head.className = "control-head";
      var label = document.createElement("label");
      label.htmlFor = "ctl-" + key;
      label.textContent = TONE_LABELS[key];
      var out = document.createElement("output");
      out.setAttribute("for", "ctl-" + key);
      var input = document.createElement("input");
      input.type = "range";
      input.id = "ctl-" + key;
      input.name = key;
      input.min = String(r.min);
      input.max = String(r.max);
      input.step = r.type === "int" ? "1" : "0.01";
      var reset = button("Reset", "reset");
      reset.setAttribute("aria-label", "Reset " + TONE_LABELS[key] + " to " + fmt(key, state.defaults[key]));
      var note = document.createElement("span");
      note.id = "ctl-note-" + key;
      note.className = "ctl-note";
      head.appendChild(label);
      head.appendChild(note);
      head.appendChild(out);
      head.appendChild(reset);
      row.appendChild(head);
      row.appendChild(input);
      host.appendChild(row);
      sliders[key] = { input: input, out: out, note: note, dim: false };
      input.addEventListener("input", function () { out.textContent = fmt(key, Number(input.value)); });
      input.addEventListener("change", function () { setParam(key, Number(input.value)); });
      reset.addEventListener("click", function () {
        input.value = String(state.defaults[key]);
        out.textContent = fmt(key, state.defaults[key]);
        setParam(key, state.defaults[key]);
      });
    });
  }

  function buildPresets() {
    var host = $("presets");
    host.textContent = "";
    state.presets.forEach(function (p) {
      var b = button(p.label, "preset");
      b.disabled = !!p.disabled;
      b.addEventListener("click", function () {
        request("POST", "/api/preset", { name: p.name }).then(function (res) {
          if (!res.ok) { say(errorOf(res), true); return; }
          state.params = res.data.params;
          syncControls();
          say("Applied preset: " + p.name, false);
        }, fail);
      });
      host.appendChild(b);
    });
  }

  function buildSlots() {
    var host = $("slots");
    host.textContent = "";
    cards = [];
    state.slotNames.forEach(function (name, idx) {
      var art = document.createElement("article");
      art.className = "slot";
      var title = document.createElement("h3");
      title.id = "slot-title-" + idx;
      title.textContent = name;
      art.setAttribute("aria-labelledby", title.id);
      var drop = document.createElement("div");
      drop.className = "drop";
      drop.tabIndex = 0;
      drop.setAttribute("role", "button");
      drop.setAttribute("aria-describedby", "slot-status-" + idx);
      drop.textContent = "Drop an audio file, or a .pth, .zip or .index RVC voice, here. Press Enter to pick a file.";
      var file = document.createElement("input");
      file.type = "file";
      file.accept = ".wav,.flac,.ogg,.mp3,.pth,.zip,.index,audio/*";
      file.id = "slot-file-" + idx;
      var fileLabel = document.createElement("label");
      fileLabel.htmlFor = file.id;
      fileLabel.textContent = "Reference audio file";
      var row = document.createElement("div");
      row.className = "row";
      var rec = button("Record 10 s", "rec");
      var sel = button("Select for manual", "sel");
      sel.setAttribute("aria-pressed", "false");
      var clr = button("Clear", "clr");
      row.appendChild(rec);
      row.appendChild(sel);
      row.appendChild(clr);
      var wave = document.createElement("canvas");
      wave.width = 240;
      wave.height = 48;
      wave.setAttribute("role", "img");
      wave.setAttribute("aria-label", name + " live waveform");
      var meter = document.createElement("div");
      meter.className = "meter";
      meter.setAttribute("role", "meter");
      meter.setAttribute("aria-label", name + " input level");
      meter.setAttribute("aria-valuemin", "0");
      meter.setAttribute("aria-valuemax", "1");
      meter.setAttribute("aria-valuenow", "0");
      var fill = document.createElement("div");
      fill.className = "fill";
      meter.appendChild(fill);
      var clip = document.createElement("span");
      clip.className = "clip";
      clip.textContent = "CLIP";
      clip.hidden = true;
      var status = document.createElement("p");
      status.className = "slot-status";
      status.id = "slot-status-" + idx;
      status.setAttribute("aria-live", "polite");
      status.textContent = "Empty";
      var kindLine = document.createElement("p");
      kindLine.className = "slot-kind";
      var kindBadge = document.createElement("span");
      kindBadge.className = "kind kind-empty";
      kindBadge.textContent = "Empty";
      var kindFacts = document.createElement("span");
      kindFacts.className = "slot-facts";
      kindLine.appendChild(kindBadge);
      kindLine.appendChild(kindFacts);
      var openBtn = button("RVC voice", "rvc-toggle");
      openBtn.id = "rvc-open-" + idx;
      openBtn.setAttribute("aria-expanded", "false");
      openBtn.setAttribute("aria-controls", "rvc-panel-" + idx);
      var panel = document.createElement("div");
      panel.id = "rvc-panel-" + idx;
      panel.className = "rvc-panel";
      panel.hidden = true;
      var rvcFileLabel = document.createElement("label");
      rvcFileLabel.htmlFor = "rvc-file-" + idx;
      rvcFileLabel.textContent = "RVC voice files (.pth, .index or .zip)";
      var rvcFile = document.createElement("input");
      rvcFile.type = "file";
      rvcFile.multiple = true;
      rvcFile.accept = ".pth,.index,.zip";
      rvcFile.id = "rvc-file-" + idx;
      var fileRow = document.createElement("div");
      fileRow.className = "rvc-row";
      fileRow.appendChild(rvcFileLabel);
      fileRow.appendChild(rvcFile);
      var linkLabel = document.createElement("label");
      linkLabel.htmlFor = "rvc-link-" + idx;
      linkLabel.textContent = "Voice link (.zip or .pth)";
      var linkInput = document.createElement("input");
      linkInput.type = "text";
      linkInput.id = "rvc-link-" + idx;
      linkInput.setAttribute("inputmode", "url");
      linkInput.setAttribute("autocomplete", "off");
      var linkGo = button("Import link", "rvc-go");
      linkGo.id = "rvc-link-go-" + idx;
      var linkRow = document.createElement("div");
      linkRow.className = "rvc-row";
      linkRow.appendChild(linkLabel);
      linkRow.appendChild(linkInput);
      linkRow.appendChild(linkGo);
      var useLabel = document.createElement("label");
      useLabel.htmlFor = "rvc-use-" + idx;
      useLabel.textContent = "Imported voice";
      var useSel = document.createElement("select");
      useSel.id = "rvc-use-" + idx;
      var useGo = button("Use this voice", "rvc-go");
      useGo.id = "rvc-use-go-" + idx;
      var useRow = document.createElement("div");
      useRow.className = "rvc-row";
      useRow.appendChild(useLabel);
      useRow.appendChild(useSel);
      useRow.appendChild(useGo);
      var delBtn = button("Delete this voice from disk", "rvc-go");
      delBtn.id = "rvc-del-" + idx;
      var delRow = document.createElement("div");
      delRow.className = "rvc-row";
      delRow.appendChild(delBtn);
      panel.appendChild(fileRow);
      panel.appendChild(linkRow);
      panel.appendChild(useRow);
      panel.appendChild(delRow);
      [title, kindLine, drop, fileLabel, file, row, openBtn, panel, wave, meter, clip, status].forEach(function (el) { art.appendChild(el); });
      host.appendChild(art);
      var card = { idx: idx, art: art, file: file, rec: rec, sel: sel, wave: wave, meter: meter, fill: fill,
                   clip: clip, status: status, kindBadge: kindBadge, kindFacts: kindFacts, openBtn: openBtn,
                   panel: panel, useSel: useSel };
      cards.push(card);
      openBtn.addEventListener("click", function () { toggleRvcPanel(card); });
      rvcFile.addEventListener("change", function () {
        if (rvcFile.files && rvcFile.files.length) { importRvc(idx, Array.from(rvcFile.files)); }
        rvcFile.value = "";
      });
      linkGo.addEventListener("click", function () { importSlotLink(idx, linkInput.value.trim()); });
      useGo.addEventListener("click", function () { useVoice(idx); });
      delBtn.addEventListener("click", function () { deleteVoice(idx); });

      drop.addEventListener("click", function () { file.click(); });
      drop.addEventListener("keydown", function (ev) {
        if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); file.click(); }
      });
      drop.addEventListener("dragover", function (ev) { ev.preventDefault(); drop.classList.add("over"); });
      drop.addEventListener("dragleave", function () { drop.classList.remove("over"); });
      drop.addEventListener("drop", function (ev) {
        ev.preventDefault();
        drop.classList.remove("over");
        if (ev.dataTransfer && ev.dataTransfer.files.length) { routeFiles(idx, Array.from(ev.dataTransfer.files)); }
      });
      file.addEventListener("change", function () {
        if (file.files && file.files.length) { routeFiles(idx, Array.from(file.files)); }
        file.value = "";
      });
      rec.addEventListener("click", function () { toggleRecord(idx); });
      sel.addEventListener("click", function () { selectManual(idx); });
      clr.addEventListener("click", function () { clearSlot(idx); });
    });
  }

  function applyCard(idx, row, note) {
    var c = cards[idx];
    if (!row) { c.status.textContent = note || ""; return; }
    state.slots[idx] = row;
    renderSlotCard(idx);
    c.art.classList.toggle("loaded", !!row.loaded);
    if (!row.loaded) { c.status.textContent = "Empty"; }
    else if (row.kind === "rvc") { c.status.textContent = "RVC voice " + row.model + " loaded."; }
    else if (row.seconds === null || row.seconds === undefined) { c.status.textContent = "Loaded (length not reported)"; }
    else { c.status.textContent = "Loaded: " + Number(row.seconds).toFixed(2) + " s"; }
  }

  function loadFile(idx, file) {
    if (!/\.(wav|flac|ogg|mp3)$/i.test(file.name)) {
      say("Drop a wav, flac, ogg or mp3 file, or a .pth, .zip or .index RVC voice file.", true);
      return;
    }
    cards[idx].status.textContent = "Loading " + file.name + " ...";
    var form = new FormData();
    form.append("file", file, file.name);
    fetch("/api/slot/" + idx, { method: "POST", body: form }).then(parseResponse).then(function (res) {
      if (!res.ok) { applyCard(idx, null, errorOf(res)); say(errorOf(res), true); return; }
      applyCard(idx, res.data.slot);
      say("Loaded " + file.name + " into " + state.slotNames[idx] + ".", false);
    }, fail);
  }

  function importUrl() {
    var idx = Number($("url-slot").value);
    var url = $("url-input").value.trim();
    if (!url) { say("Paste a link to an audio file first.", true); return; }
    var path = null;
    try { path = new URL(url).pathname.toLowerCase(); } catch (err) { say("That is not a valid link.", true); return; }
    if (/\.(pth|zip)$/.test(path)) { sendRvc(idx, "/api/slot/" + idx + "/rvc/link", jsonInit({ url: url }), "Importing link ..."); return; }
    say("Downloading ...", false);
    fetch("/api/slot/" + idx + "/url", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url: url }) }).then(parseResponse).then(function (res) {
      if (!res.ok) { say(errorOf(res), true); return; }
      applyCard(idx, res.data.slot);
      say("Loaded the link into " + state.slotNames[idx] + ".", false);
    }, fail);
  }

  var webVoicesReady = false;
  function setupWebVoices() {
    var sel = $("url-slot");
    sel.textContent = "";
    state.slotNames.forEach(function (name, idx) {
      var opt = document.createElement("option");
      opt.value = String(idx);
      opt.textContent = name;
      sel.appendChild(opt);
    });
    if (webVoicesReady) { return; }
    webVoicesReady = true;
    $("url-go").addEventListener("click", importUrl);
    $("library-open").addEventListener("click", function () { fetch("/api/open_voice_library", { method: "POST" }).then(parseResponse).then(function (res) { if (!res.ok) { say(errorOf(res), true); } }, fail); });
  }

  function setRecording(idx, on) {
    var c = cards[idx];
    c.rec.textContent = on ? "Stop" : "Record 10 s";
    c.rec.setAttribute("aria-pressed", on ? "true" : "false");
  }

  function toggleRecord(idx) {
    if (recordSlot === idx) { stopRecord(); return; }
    if (recordSlot !== null) { say("Stop the running recording first.", true); return; }
    request("POST", "/api/slot/" + idx + "/record/start").then(function (res) {
      if (!res.ok) { say(errorOf(res), true); return; }
      recordSlot = idx;
      recordStarted = Date.now();
      setRecording(idx, true);
      cards[idx].status.textContent = "Recording: 0.0 s";
      tickTimer = setInterval(function () {
        cards[idx].status.textContent = "Recording: " + ((Date.now() - recordStarted) / 1000).toFixed(1) + " s";
      }, 250);
      recordTimer = setTimeout(stopRecord, state.recordMax * 1000);
    }, fail);
  }

  function stopRecord() {
    var idx = recordSlot;
    if (idx === null) { return; }
    clearTimeout(recordTimer);
    clearInterval(tickTimer);
    recordSlot = null;
    request("POST", "/api/slot/" + idx + "/record/stop").then(function (res) {
      setRecording(idx, false);
      if (!res.ok) { applyCard(idx, null, errorOf(res)); say(errorOf(res), true); return; }
      applyCard(idx, res.data.slot);
      say("Recorded into " + state.slotNames[idx] + ".", false);
    }, function (err) { setRecording(idx, false); fail(err); });
  }

  function clearSlot(idx) {
    request("DELETE", "/api/slot/" + idx).then(function (res) {
      if (!res.ok) { say(errorOf(res), true); return; }
      applyCard(idx, res.data.slot);
      say(state.slotNames[idx] + " cleared.", false);
    }, fail);
  }

  function selectManual(idx) {
    request("POST", "/api/manual_slot", { idx: idx }).then(function (res) {
      if (!res.ok) { say(errorOf(res), true); return; }
      state.params.manual_slot = idx;
      state.params.emotion_mode = "manual";
      syncControls();
      say("Manual slot: " + state.slotNames[idx] + ".", false);
    }, fail);
  }

  function syncControls() {
    var p = state.params || {};
    Object.keys(sliders).forEach(function (key) {
      var s = sliders[key];
      if (document.activeElement !== s.input && p[key] !== undefined) { s.input.value = String(p[key]); }
      s.out.textContent = fmt(key, Number(s.input.value));
    });
    BOOL_KEYS.forEach(function (key) {
      var el = $(key);
      if (document.activeElement !== el) { el.checked = !!p[key]; }
    });
    var manual = p.emotion_mode === "manual";
    $("emo-auto").checked = !manual;
    $("emo-manual").checked = manual;
    cards.forEach(function (c) {
      c.sel.setAttribute("aria-pressed", String(manual && p.manual_slot === c.idx));
    });
  }

  function fmtMs(v) { return v === null || v === undefined ? "n/a" : Math.round(v) + " ms"; }

  function balancerText(b) {
    if (!b || typeof b !== "object") { return "n/a"; }
    var parts = [];
    Object.keys(b).forEach(function (k) {
      var v = b[k];
      if (v === null || typeof v === "object") { return; }
      parts.push(k + "=" + String(v));
    });
    return parts.length ? parts.join(", ") : "n/a";
  }

  function renderHeader(tel) {
    var input = tel.input || {};
    var output = tel.output || {};
    var status = tel.running ? "running" : "stopped";
    if (tel.error) { status += " (" + tel.error + ")"; }
    setText("st-pipeline", status);
    setText("st-input", input.mode ? input.mode + " / " + (input.device || "n/a") : "not open");
    setText("st-output", output.device ? output.device : "not open");
    setText("st-latency", output.e2e_ms_last === null || output.e2e_ms_last === undefined ? "n/a"
      : "last " + fmtMs(output.e2e_ms_last) + ", mean " + fmtMs(output.e2e_ms_mean) + ", max " + fmtMs(output.e2e_ms_max));
    setText("st-underruns", output.underruns === null || output.underruns === undefined ? "n/a" : String(output.underruns));
    setText("st-balancer", balancerText(output.balancer));
  }

  function drawWave(canvas, samples) {
    var ctx = canvas.getContext("2d");
    var w = canvas.width;
    var h = canvas.height;
    ctx.fillStyle = "#0b0d11";
    ctx.fillRect(0, 0, w, h);
    if (!samples || samples.length < 2) { return; }
    ctx.strokeStyle = "#6aa9ff";
    ctx.beginPath();
    for (var i = 0; i < samples.length; i++) {
      var x = (i / (samples.length - 1)) * w;
      var v = Math.max(-1, Math.min(1, samples[i]));
      var y = (0.5 - v * 0.5) * h;
      if (i === 0) { ctx.moveTo(x, y); } else { ctx.lineTo(x, y); }
    }
    ctx.stroke();
  }

  function drawMeter(card, peak, clipping) {
    var level = Math.max(0, Math.min(1, Number(peak) || 0));
    card.fill.style.width = (level * 100).toFixed(1) + "%";
    card.meter.setAttribute("aria-valuenow", level.toFixed(2));
    card.meter.classList.toggle("clipping", !!clipping);
    card.clip.hidden = !clipping;
  }

  function onLive(msg) {
    var tel = msg.telemetry || {};
    if (tel.params) { state.params = tel.params; }
    renderHeader(tel);
    applyEngine(tel);
    syncControls();
    cards.forEach(function (c) {
      var on = msg.recording === c.idx && !!msg.levels;
      drawWave(c.wave, on ? msg.levels.waveform : null);
      drawMeter(c, on ? msg.levels.peak : 0, on ? msg.levels.clipping : false);
    });
  }

  function jsonInit(body) {
    return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  }

  function findModel(name) {
    var models = state.rvc.models || [];
    for (var i = 0; i < models.length; i++) {
      if (models[i].name === name) { return models[i]; }
    }
    return null;
  }

  function voiceFacts(model) {
    var parts = [model.version + " " + (Number(model.sr) / 1000) + "k"];
    if (model.has_index) { parts.push("index"); }
    return parts.join(", ");
  }

  function renderSlotCard(idx) {
    var c = cards[idx];
    if (!c) { return; }
    var row = state.slots[idx];
    var kind = row && row.loaded ? row.kind : null;
    var model = kind === "rvc" ? findModel(row.model) : null;
    c.kindBadge.className = "kind kind-" + (kind || "empty");
    c.kindBadge.textContent = kind === "rvc" ? "RVC voice" : kind === "clip" ? "Clip" : "Empty";
    c.kindFacts.textContent = kind === "rvc" ? (row.model || "unknown voice") + (model ? ", " + voiceFacts(model) : "") : "";
    var prev = c.useSel.value;
    c.useSel.textContent = "";
    var none = document.createElement("option");
    none.value = "";
    none.textContent = "(choose a voice)";
    c.useSel.appendChild(none);
    (state.rvc.models || []).forEach(function (m) {
      var opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = m.name + " (" + voiceFacts(m) + ")";
      c.useSel.appendChild(opt);
    });
    c.useSel.value = prev;
    if (c.useSel.selectedIndex < 0) { c.useSel.selectedIndex = 0; }
  }

  function renderAllSlots() {
    cards.forEach(function (c) { renderSlotCard(c.idx); });
  }

  function renderWeights() {
    var w = state.rvc.weights || {};
    var text = "RVC base models: status unknown";
    if (w.state === "ready") { text = "RVC base models: ready"; }
    else if (w.state === "missing") { text = "RVC base models: missing (about 370 MB, fetched the first time an RVC voice is used)"; }
    else if (w.state === "downloading") {
      var pct = w.total > 0 ? Math.floor((w.done * 100) / w.total) : 0;
      var mb = Math.round((w.total || 0) / 1048576);
      text = "RVC base models: downloading " + pct + "% of " + mb + " MB" + (w.file ? " (" + w.file + ")" : "");
    }
    else if (w.state === "error") { text = "RVC base models: " + (w.message || "error"); }
    setText("rvc-weights", text);
  }

  function weightsDownloading() {
    return !!state.rvc.weights && state.rvc.weights.state === "downloading";
  }

  function startWeightsPoll() {
    var token = ++weightsToken;
    setTimeout(function () {
      if (token !== weightsToken) { return; }
      request("GET", "/api/rvc/models").then(function (res) {
        if (token !== weightsToken) { return; }
        if (!res.ok) { say(errorOf(res), true); return; }
        state.rvc = { models: res.data.models, weights: res.data.weights };
        renderAllSlots();
        renderWeights();
        if (weightsDownloading()) { startWeightsPoll(); }
      }, fail);
    }, WEIGHTS_POLL_MS);
  }

  function refreshState() {
    request("GET", "/api/state").then(function (res) {
      if (!res.ok) { say(errorOf(res), true); return; }
      state.rvc = res.data.rvc;
      res.data.slots.forEach(function (row) { state.slots[row.idx] = row; });
      renderAllSlots();
      renderWeights();
      if (weightsDownloading()) { startWeightsPoll(); }
    }, fail);
  }

  function sendRvc(idx, url, init, progress) {
    var card = cards[idx];
    var before = card.status.textContent;
    card.status.textContent = progress;
    fetch(url, init).then(parseResponse).then(function (res) {
      refreshState();
      if (!res.ok) { card.status.textContent = before; say(errorOf(res), true); return; }
      applyCard(idx, res.data.slot);
      say("Loaded RVC voice " + res.data.model.name + " into " + state.slotNames[idx] + ".", false);
    }, function (err) { card.status.textContent = before; fail(err); });
  }

  function routeFiles(idx, files) {
    if (!files.length) { return; }
    var audio = files.filter(function (f) { return /\.(wav|flac|ogg|mp3)$/i.test(f.name); });
    var rvc = files.filter(function (f) { return /\.(pth|zip|index)$/i.test(f.name); });
    if (rvc.length && audio.length) { say("Drop audio files or RVC voice files on a slot, not both.", true); return; }
    if (rvc.length) { importRvc(idx, rvc); return; }
    loadFile(idx, files[0]);
  }

  function importRvc(idx, files) {
    var zips = files.filter(function (f) { return /\.zip$/i.test(f.name); });
    var pths = files.filter(function (f) { return /\.pth$/i.test(f.name); });
    var indexes = files.filter(function (f) { return /\.index$/i.test(f.name); });
    if (zips.length > 1 || pths.length > 1 || indexes.length > 1) { say("Drop one .zip, or one .pth with its .index.", true); return; }
    if (zips.length && (pths.length || indexes.length)) { say("A .zip already holds its .pth and .index. Drop the .zip on its own.", true); return; }
    if (!zips.length && !pths.length) { say("An .index file needs its .pth file. Drop both together.", true); return; }
    var form = new FormData();
    if (zips.length) { form.append("zip", zips[0], zips[0].name); }
    if (pths.length) { form.append("pth", pths[0], pths[0].name); }
    if (indexes.length) { form.append("index", indexes[0], indexes[0].name); }
    var label = (zips.length ? zips[0] : pths[0]).name;
    sendRvc(idx, "/api/slot/" + idx + "/rvc/import", { method: "POST", body: form }, "Importing " + label + " ...");
  }

  function importSlotLink(idx, link) {
    if (!link) { say("Paste a link to a .zip or .pth file first.", true); return; }
    var path = null;
    try { path = new URL(link).pathname.toLowerCase(); } catch (err) { path = null; }
    if (path === null || !/\.(pth|zip)$/.test(path)) { say("Paste a link to a .zip or .pth file for an RVC voice.", true); return; }
    sendRvc(idx, "/api/slot/" + idx + "/rvc/link", jsonInit({ url: link }), "Importing link ...");
  }

  function useVoice(idx) {
    var name = cards[idx].useSel.value;
    if (!name) { say("Choose an imported voice first.", true); return; }
    sendRvc(idx, "/api/slot/" + idx + "/rvc/use", jsonInit({ name: name }), "Loading " + name + " ...");
  }

  function deleteVoice(idx) {
    var row = state.slots[idx];
    var name = (row && row.model) || cards[idx].useSel.value;
    if (!name) { say("Choose a voice to delete first.", true); return; }
    if (!window.confirm("Delete the voice " + name + " from disk? Any slot that uses it will be cleared.")) { return; }
    request("DELETE", "/api/rvc/models/" + encodeURIComponent(name)).then(function (res) {
      if (!res.ok) { say(errorOf(res), true); return; }
      (res.data.cleared_slots || []).forEach(function (i) { applyCard(i, null, "Empty"); });
      say("Deleted voice " + name + ".", false);
      refreshState();
    }, fail);
  }

  function toggleRvcPanel(card) {
    var show = card.panel.hidden;
    card.panel.hidden = !show;
    card.openBtn.setAttribute("aria-expanded", show ? "true" : "false");
    if (!show && card.panel.contains(document.activeElement)) { card.openBtn.focus(); }
  }

  var ENGINE_TEXT = { clip: "Seed-VC (reference clip)", rvc: "RVC voice" };

  function renderEngine(eng) {
    var text = ENGINE_TEXT[eng.kind] || "none";
    if (eng.switching) { text += " (switching)"; }
    if (typeof eng.waiting === "string" && eng.waiting) { text += ", " + eng.waiting; }
    if (text !== engineText) { engineText = text; setText("st-engine", text); }
    var err = typeof eng.error === "string" && eng.error ? eng.error : null;
    if (err !== null && err !== lastEngineError) { say(err, true); }
    lastEngineError = err;
  }

  function applyEngineControls(eng) {
    var rvc = !!eng && eng.kind === "rvc";
    var unused = rvc && Array.isArray(eng.unused_controls) ? eng.unused_controls : [];
    Object.keys(sliders).forEach(function (key) {
      var s = sliders[key];
      var off = unused.indexOf(key) >= 0;
      if (s.dim === off) { return; }
      s.dim = off;
      s.input.disabled = off;
      s.note.textContent = off ? "not used by RVC voices" : "";
      if (off) { s.input.setAttribute("aria-describedby", s.note.id); }
      else { s.input.removeAttribute("aria-describedby"); }
    });
  }

  function applyEngine(tel) {
    if (!tel.engine) { return; }
    renderEngine(tel.engine);
    applyEngineControls(tel.engine);
  }

  function applyState(doc) {
    state.ranges = doc.ranges;
    state.defaults = doc.defaults;
    state.params = doc.params;
    state.slotNames = doc.slot_names;
    state.presets = doc.presets;
    state.recordMax = doc.recording.max_seconds;
    state.rvc = doc.rvc;
    state.slots = [];
    buildSliders();
    buildSlots();
    setupWebVoices();
    buildPresets();
    doc.slots.forEach(function (row) { applyCard(row.idx, row); });
    renderHeader(doc.telemetry || {});
    syncControls();
    renderWeights();
    if (weightsDownloading()) { startWeightsPoll(); }
    applyEngine(doc.telemetry || {});
  }

  function connect() {
    var scheme = window.location.protocol === "https:" ? "wss://" : "ws://";
    var socket = new WebSocket(scheme + window.location.host + "/ws");
    socket.onmessage = function (ev) {
      var msg;
      try { msg = JSON.parse(ev.data); } catch (err) { return; }
      onLive(msg);
    };
    socket.onclose = function () {
      setText("st-pipeline", "telemetry disconnected, retrying");
      setTimeout(connect, 1000);
    };
  }

  BOOL_KEYS.forEach(function (key) {
    $(key).addEventListener("change", function () { setParam(key, $(key).checked); });
  });
  document.querySelectorAll('input[name="emotion_mode"]').forEach(function (radio) {
    radio.addEventListener("change", function () {
      if (radio.checked) { setParam("emotion_mode", radio.value); }
    });
  });

  request("GET", "/api/state").then(function (res) {
    if (!res.ok) { throw new Error(errorOf(res)); }
    applyState(res.data);
    connect();
  }).catch(function (err) { say("could not load /api/state: " + err.message, true); });
})();
