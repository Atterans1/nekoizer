(function () {
  "use strict";

  /* ---------------------------------------------------------------- PUBLIC */

  var State = { params: {}, ranges: {}, running: false, references: {}, presets: [] };
  var Controls = {};
  var Socket = null;
  var ReconnectDelay = 500;
  var Pending = {};
  var FlushDelayMs = 30;
  var FlushScheduled = false;

  function Init() {
    BuildControls();
    BindDevices();
    BindRail();
    BindPreset();
    BindVoiceGen();
    document.addEventListener("dragover", function (e) { e.preventDefault(); });
    document.addEventListener("drop", function (e) { e.preventDefault(); });
    RefreshState().then(LoadDevices);
    Connect();
  }

  /* --------------------------------------------------------------- PRIVATE */

  function $(selector) { return document.querySelector(selector); }

  function Api(method, url, body) {
    var options = { method: method };
    if (body instanceof FormData) {
      options.body = body;
    } else if (body !== undefined) {
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify(body);
    }
    return fetch(url, options).then(function (response) {
      return response.json().catch(function () { return {}; }).then(function (data) {
        if (!response.ok) { throw new Error(data.error || ("request failed (" + response.status + ")")); }
        return data;
      });
    });
  }

  function SetText(selector, text) { $(selector).textContent = text; }

  function ShowError(selector, message) { SetText(selector, message || ""); }

  function Banner(message) {
    var el = $("#banner");
    el.hidden = !message;
    el.textContent = message || "";
  }

  function Fmt(value, digits, unit) {
    if (typeof value !== "number" || !isFinite(value)) { return "-"; }
    return value.toFixed(digits) + (unit ? " " + unit : "");
  }

  /* ---- state + controls */

  function RefreshState() {
    return Api("GET", "/api/state").then(function (data) {
      State.params = data.params;
      State.ranges = data.ranges;
      State.running = data.running;
      State.references[0] = data.reference;
      State.presets = data.presets;
      Banner("");
      ApplyRanges();
      for (var key in State.params) { SetControlValue(key, State.params[key]); }
      RenderRunState();
      RenderRail();
    }).catch(function (e) { Banner("Cannot reach server: " + e.message); });
  }

  function BuildControls() {
    var nodes = document.querySelectorAll(".ctl[data-key]");
    Array.prototype.forEach.call(nodes, function (node) {
      var key = node.dataset.key;
      var line = document.createElement("div");
      line.className = "line";
      var label = document.createElement("span");
      label.className = "label";
      label.textContent = node.dataset.label;
      line.appendChild(label);
      var entry = { node: node, key: key, unit: node.dataset.unit || "", signed: !!node.dataset.signed };
      var err = document.createElement("div");
      err.className = "err";
      entry.err = err;
      var value = document.createElement("span");
      value.className = "value";
      entry.value = value;
      var input = document.createElement("input");
      entry.input = input;
      entry.isBool = false;
      line.appendChild(input);
      line.appendChild(value);
      node.appendChild(line);
      node.appendChild(err);
      Controls[key] = entry;
      input.addEventListener("input", function () { OnControlInput(entry); });
      input.addEventListener("change", function () { OnControlInput(entry); });
    });
  }

  function ApplyRanges() {
    for (var key in Controls) {
      var entry = Controls[key];
      var range = State.ranges[key];
      if (!range) { entry.node.hidden = true; continue; }
      entry.node.hidden = false;
      if (range.type === "bool") { MakeToggle(entry); continue; }
      entry.input.type = "range";
      entry.input.min = range.min;
      entry.input.max = range.max;
      entry.input.step = range.type === "int" ? 1 : 0.01;
      entry.isBool = false;
    }
  }

  function MakeToggle(entry) {
    if (entry.isBool) { return; }
    var wrap = document.createElement("span");
    wrap.className = "switch";
    var track = document.createElement("span");
    track.className = "track";
    var old = entry.input;
    var input = document.createElement("input");
    input.type = "checkbox";
    wrap.appendChild(input);
    wrap.appendChild(track);
    old.parentNode.replaceChild(wrap, old);
    entry.input = input;
    entry.isBool = true;
    input.addEventListener("change", function () { OnControlInput(entry); });
    entry.value.style.visibility = "hidden";
  }

  function ReadControl(entry) {
    if (entry.isBool) { return entry.input.checked; }
    var range = State.ranges[entry.key];
    var n = Number(entry.input.value);
    return range && range.type === "int" ? Math.round(n) : n;
  }

  function FormatValue(entry, value) {
    var range = State.ranges[entry.key] || {};
    var text = range.type === "int" ? String(value) : Number(value).toFixed(2);
    if (entry.signed && value > 0) { text = "+" + text; }
    return entry.unit ? text + " " + entry.unit : text;
  }

  function SetControlValue(key, value) {
    var entry = Controls[key];
    if (!entry || !State.ranges[key]) { return; }
    if (entry.isBool) {
      entry.input.checked = !!value;
    } else {
      entry.input.value = value;
      entry.value.textContent = FormatValue(entry, value);
    }
  }

  function OnControlInput(entry) {
    var value = ReadControl(entry);
    if (!entry.isBool) { entry.value.textContent = FormatValue(entry, value); }
    Pending[entry.key] = value;
    if (!FlushScheduled) {
      FlushScheduled = true;
      setTimeout(FlushPending, FlushDelayMs);
    }
  }

  function FlushPending() {
    FlushScheduled = false;
    var batch = Pending;
    Pending = {};
    for (var key in batch) {
      if (!Socket || Socket.readyState !== WebSocket.OPEN) {
        ShowControlError(key, "not connected to server");
        continue;
      }
      Socket.send(JSON.stringify({ type: "set", key: key, value: batch[key] }));
    }
  }

  function ShowControlError(key, message) {
    var entry = key && Controls[key];
    if (entry) { entry.err.textContent = message; } else { Banner(message); }
  }

  /* ---- websocket */

  function Connect() {
    var scheme = location.protocol === "https:" ? "wss://" : "ws://";
    Socket = new WebSocket(scheme + location.host + "/ws");
    Socket.onopen = function () {
      ReconnectDelay = 500;
      SetConn(true);
      RefreshState();
    };
    Socket.onclose = function () {
      SetConn(false);
      setTimeout(Connect, ReconnectDelay);
      ReconnectDelay = Math.min(ReconnectDelay * 2, 5000);
    };
    Socket.onerror = function () { Socket.close(); };
    Socket.onmessage = function (event) { OnMessage(JSON.parse(event.data)); };
  }

  function SetConn(up) {
    $("#conn").dataset.state = up ? "up" : "down";
    SetText("#conn-text", up ? "connected" : "reconnecting");
  }

  function OnMessage(msg) {
    if (msg.type === "telemetry") { RenderTelemetry(msg); }
    else if (msg.type === "ack") { OnAck(msg); }
    else if (msg.type === "error") { ShowControlError(msg.key, msg.message); }
  }

  function OnAck(msg) {
    State.params[msg.key] = msg.value;
    var entry = Controls[msg.key];
    if (entry) { entry.err.textContent = ""; }
    var editing = entry && entry.input === document.activeElement;
    if (!Pending.hasOwnProperty(msg.key) && !editing) { SetControlValue(msg.key, msg.value); }
    if (msg.key === "chunk_ms") { RefreshState(); }
  }

  /* ---- telemetry */

  var DbFloor = -60;

  function RenderMeter(fill, label, db) {
    var ok = typeof db === "number" && isFinite(db);
    var pct = ok ? Math.max(0, Math.min(100, (db - DbFloor) / -DbFloor * 100)) : 0;
    $(fill).style.width = pct + "%";
    SetText(label, ok ? db.toFixed(1) + " dB" : "-");
  }

  function RenderTelemetry(t) {
    RenderMeter("#in-fill", "#in-db", t.in_db);
    RenderMeter("#out-fill", "#out-db", t.out_db);
    if (typeof t.running === "boolean" && t.running !== State.running) {
      State.running = t.running;
      RenderRunState();
    }
    SetText("#p-e2e", Fmt(t.e2e_latency_ms, 0, "ms"));
    SetText("#p-inf", Fmt(t.inference_ms, 0, "ms") + " / " + Fmt(t.chunk_ms, 0, "ms"));
    $("#p-rt").dataset.ok = State.running ? String(!!t.realtime_ok) : "unknown";
    var gpu = Fmt(t.gpu_util_pct, 0, "%");
    if (typeof t.gpu_mem_mb === "number") { gpu += " · " + Fmt(t.gpu_mem_mb, 0, "MB"); }
    SetText("#p-gpu", gpu);
    SetText("#p-cpu", Fmt(t.cpu_pct, 0, "%"));
    var backend = [t.backend, t.precision, t.device].filter(Boolean).join(" · ");
    SetText("#p-backend", backend || "-");
    SetText("#p-drop", (t.underruns == null ? "-" : t.underruns + " underruns") + (t.vad_skipped_chunks == null ? "" : " · " + t.vad_skipped_chunks + " VAD-skipped"));
  }

  /* ---- devices + start/stop */

  function Populate(select, devices, prefer) {
    select.innerHTML = "";
    var chosen = -1;
    devices.forEach(function (d, i) {
      var option = document.createElement("option");
      option.value = d.id;
      option.textContent = d.name + " [" + d.hostapi + "]";
      select.appendChild(option);
      if (chosen < 0 && prefer && d.name.indexOf(prefer) !== -1) { chosen = i; }
    });
    if (chosen >= 0) { select.selectedIndex = chosen; }
  }

  function LoadDevices() {
    return Api("GET", "/api/devices").then(function (data) {
      Populate($("#input-device"), data.inputs, null);
      Populate($("#output-device"), data.outputs, null);   // server sorts the default output first
      ShowError("#start-err", "");
    }).catch(function (e) { ShowError("#start-err", "device list failed: " + e.message); });
  }

  function BindDevices() {
    $("#start").addEventListener("click", function () {
      var button = $("#start");
      button.disabled = true;
      var call = State.running
        ? Api("POST", "/api/stop")
        : Api("POST", "/api/start", {
            input_id: parseInt($("#input-device").value, 10),
            output_id: parseInt($("#output-device").value, 10)
          });
      call.then(function (data) {
        State.running = data.running;
        ShowError("#start-err", "");
      }).catch(function (e) {
        ShowError("#start-err", e.message);
      }).then(function () {
        button.disabled = false;
        RenderRunState();
      });
    });
  }

  function RenderRunState() {
    var running = State.running;
    var button = $("#start");
    button.textContent = running ? "Stop" : "Start";
    button.dataset.running = String(running);
    var tag = $("#runstate");
    tag.textContent = running ? "running" : "stopped";
    tag.dataset.running = String(running);
    $("#input-device").disabled = running;
    $("#output-device").disabled = running;
    RenderVoiceGen();
  }

  /* ---- voice slots + upload */

  function RenderRail() {
    var holder = $("#slots");
    holder.innerHTML = "";
    for (var i = 0; i < 10; i++) {
      var ref = State.references[i];
      var card = document.createElement("div");
      card.className = "slot" + (ref ? " active" : "");
      card.dataset.slot = i;
      card.setAttribute("role", "button");
      card.setAttribute("tabindex", "0");
      var label = document.createElement("span");
      label.className = "name";
      label.textContent = "Slot " + (i + 1);
      var info = document.createElement("span");
      info.className = "info";
      info.textContent = ref ? ref.name + " · " + Fmt(ref.duration_s, 1, "s") + " · " + (ref.sample_rate || "-") + " Hz" : "empty";
      var del = document.createElement("button");
      del.className = "del";
      del.type = "button";
      del.title = "Clear slot";
      del.textContent = "×";
      card.appendChild(label);
      card.appendChild(info);
      card.appendChild(del);
      card.addEventListener("click", function () { PickSlot(this.dataset.slot); });
      card.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { PickSlot(this.dataset.slot); }
      });
      del.addEventListener("click", function (e) {
        e.stopPropagation();
        ClearSlot(this.closest(".slot").dataset.slot);
      });
      card.addEventListener("dragover", function (e) { e.preventDefault(); this.classList.add("over"); });
      card.addEventListener("dragleave", function () { this.classList.remove("over"); });
      card.addEventListener("drop", function (e) {
        e.preventDefault();
        this.classList.remove("over");
        if (e.dataTransfer.files.length) { Upload(e.dataTransfer.files[0], this.dataset.slot); }
      });
      holder.appendChild(card);
    }
  }

  function LoadPreset(name) {
    Api("POST", "/api/presets/load", { name: name }).then(function (ref) {
      State.references[0] = ref;
      ShowError("#upload-err", "");
      RenderRail();
    }).catch(function (e) { ShowError("#upload-err", e.message); });
  }

  function DeletePreset(name) {
    if (!window.confirm("Delete preset '" + name + "'?")) { return; }
    Api("DELETE", "/api/presets/" + encodeURIComponent(name)).then(function (data) {
      State.presets = data.presets;
      ShowError("#upload-err", "");
      RenderRail();
    }).catch(function (e) { ShowError("#upload-err", e.message); });
  }

  function Upload(file, slot) {
    var form = new FormData();
    form.append("file", file, file.name);
    form.append("slot", slot);
    ShowError("#upload-err", "");
    Api("POST", "/api/reference", form).then(function (ref) {
      State.references[slot] = ref;
      RenderRail();
    }).catch(function (e) { ShowError("#upload-err", e.message); });
  }

  function PickSlot(slot) {
    var input = $("#file");
    input.dataset.slot = slot;
    input.click();
  }

  function ClearSlot(slot) {
    Api("DELETE", "/api/reference/" + encodeURIComponent(slot)).then(function (data) {
      State.references[slot] = null;
      RenderRail();
    }).catch(function (e) { ShowError("#upload-err", e.message); });
  }

  
  function BindRail() {
    var input = $("#file");
    input.addEventListener("change", function () {
      if (input.files.length) { Upload(input.files[0], input.dataset.slot || 0); }
      input.value = "";
    });
    var urlGo = $("#url-go");
    if (urlGo) {
      urlGo.addEventListener("click", function () {
        var url = $("#url-input").value.trim();
        if (!url) { ShowError("#url-err", "Please enter a URL."); return; }
        ShowError("#url-err", "");
        ImportFromUrl(url);
      });
    }
  }

  function ImportFromUrl(url) {
    ShowError("#url-err", "");
    Api("POST", "/api/slot/0/url", JSON.stringify({ url: url })).then(function (ref) {
      State.references[0] = ref;
      RenderRail();
    }).catch(function (e) { ShowError("#url-err", e.message); });
  }

  function BindPreset() {
    $("#save-preset").addEventListener("click", function () {
      var name = $("#preset-name").value.trim();
      Api("POST", "/api/presets/save", { name: name }).then(function (data) {
        State.presets = data.presets;
        State.references[0] = data.reference;
        ShowError("#preset-err", "");
        RenderRail();
      }).catch(function (e) { ShowError("#preset-err", e.message); });
    });
  }

  /* ---- voice creation */

  var EngineRunningMessage = "Stop the live engine first so CPU inference cannot cause audio dropouts";
  var VoiceGen = { jobId: null, status: "idle", timer: null, accepted: null };

  function VoiceGenBusy() { return VoiceGen.status === "queued" || VoiceGen.status === "running"; }

  function RenderVoiceGen() {
    var busy = VoiceGenBusy();
    var blocked = State.running;
    var ready = VoiceGen.status === "ready";
    $("#vg-text").disabled = busy || blocked;
    $("#vg-create").disabled = busy || blocked;
    $("#vg-create").textContent = busy ? "Creating..." : "Create voice";
    $("#vg-accept").disabled = blocked || !ready;
    $("#vg-reject").disabled = blocked || !ready;
    if (blocked) { ShowError("#vg-err", EngineRunningMessage); }
    else if ($("#vg-err").textContent === EngineRunningMessage) { ShowError("#vg-err", ""); }
  }

  function ResetVoiceGenResult() {
    $("#vg-result").hidden = true;
    $("#vg-download").hidden = true;
    var audio = $("#vg-audio");
    audio.pause();
    audio.removeAttribute("src");
    VoiceGen.accepted = null;
  }

  function BeginVoiceGen(call) {
    ShowError("#vg-err", "");
    ResetVoiceGenResult();
    var log = $("#vg-log");
    log.hidden = false;
    log.textContent = "";
    VoiceGen.status = "queued";
    RenderVoiceGen();
    call.then(function (data) {
      VoiceGen.jobId = data.job_id;
      PollVoiceGen();
    }).catch(function (e) {
      VoiceGen.status = "idle";
      log.hidden = true;
      ShowError("#vg-err", e.message);
      RenderVoiceGen();
    });
  }

  function PollVoiceGen() {
    var jobId = VoiceGen.jobId;
    Api("GET", "/api/voicegen/" + jobId).then(function (job) {
      if (jobId !== VoiceGen.jobId) { return; }
      VoiceGen.status = job.status;
      var log = $("#vg-log");
      log.textContent = job.log.join("\n");
      log.scrollTop = log.scrollHeight;
      if (job.status === "ready") { ShowVoiceGenResult(job); }
      else if (job.status === "failed") { ShowError("#vg-err", job.error || "voice creation failed"); }
      else { VoiceGen.timer = setTimeout(PollVoiceGen, 1000); }
      RenderVoiceGen();
    }).catch(function (e) {
      VoiceGen.status = "failed";
      ShowError("#vg-err", e.message);
      RenderVoiceGen();
    });
  }

  function SafeHttpUrl(url) { return typeof url === "string" && /^https?:\/\//i.test(url) ? url : null; }

  function ShowVoiceGenResult(job) {
    var result = job.result || {};
    $("#vg-audio").src = "/api/voicegen/" + job.job_id + "/clip.wav";
    var line = $("#vg-source");
    line.textContent = "";
    line.appendChild(document.createTextNode("Source: "));
    var href = SafeHttpUrl(result.source_url);
    if (href) {
      var link = document.createElement("a");
      link.href = href;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = result.source_title || href;
      line.appendChild(link);
    } else {
      line.appendChild(document.createTextNode(result.source_title || "unknown"));
    }
    line.appendChild(document.createTextNode(" · License: " + (result.license || "unknown")));
    $("#vg-result").hidden = false;
    var box = $("#vg-text");
    box.value = "";
    box.placeholder = "What to change? e.g. deeper, slower, then press Not quite";
  }

  function BindVoiceGen() {
    $("#vg-create").addEventListener("click", function () {
      BeginVoiceGen(Api("POST", "/api/voicegen/start", { prompt: $("#vg-text").value.trim() }));
    });
    $("#vg-reject").addEventListener("click", function () {
      var feedback = $("#vg-text").value.trim();
      BeginVoiceGen(Api("POST", "/api/voicegen/" + VoiceGen.jobId + "/reject", { feedback: feedback }));
    });
    $("#vg-accept").addEventListener("click", function () {
      var name = window.prompt("Preset name (letters, digits, _ and -, up to 40):");
      if (name === null) { return; }
      name = name.trim();
      Api("POST", "/api/voicegen/" + VoiceGen.jobId + "/accept", { name: name }).then(function (data) {
        State.presets = data.presets;
        State.references[0] = data.reference;
        VoiceGen.accepted = name;
        var link = $("#vg-download");
        link.href = "/api/presets/" + encodeURIComponent(name) + "/download";
        link.download = name + ".wav";
        link.hidden = false;
        ShowError("#vg-err", "");
        RenderRail();
      }).catch(function (e) { ShowError("#vg-err", e.message); });
    });
    RenderVoiceGen();
  }

  Init();
})();
