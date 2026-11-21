(function () {
  "use strict";
  var MIMI = { thinking: "Mimi is thinking", talking: "Mimi is talking", sleeping: "Mimi is sleeping", error: "Mimi is worried" };
  var img = document.getElementById("mimi");
  var pipeline = document.getElementById("st-pipeline");
  var msg = document.getElementById("msg");
  function mood() {
    var p = pipeline.textContent;
    if (msg.classList.contains("error") || p.indexOf("(") !== -1 || p.indexOf("disconnected") !== -1) { return "error"; }
    if (p === "running") { return "talking"; }
    if (p === "stopped") { return "sleeping"; }
    return "thinking";
  }
  function render() {
    var m = mood();
    var src = "/static/art/mimi_" + m + ".png";
    if (img.getAttribute("src") !== src) { img.setAttribute("src", src); img.setAttribute("alt", MIMI[m]); }
  }
  new MutationObserver(render).observe(pipeline, { childList: true, characterData: true, subtree: true });
  new MutationObserver(render).observe(msg, { attributes: true, attributeFilter: ["class"] });
  render();
})();
