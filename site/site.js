(function () {
  "use strict";
  var c = window.SITE_CONFIG;
  document.querySelectorAll("[data-donate]").forEach(function (a) { a.href = c.donateUrl; });
  document.querySelectorAll("[data-download]").forEach(function (a) { a.href = c.msiUrl; });
  document.querySelectorAll("[data-version]").forEach(function (s) { s.textContent = c.version; });
})();

(function () {
  "use strict";
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) { return; }
  var mimi = document.querySelector(".top img");
  if (!mimi) { return; }
  mimi.addEventListener("mouseenter", function () { mimi.src = "art/mimi_talking.png"; });
  mimi.addEventListener("mouseleave", function () { mimi.src = "art/mimi_idle.png"; });
  var glyphs = ["\u2726", "\u2665", "\u2727"], colors = ["#FF8FC7", "#7FE5C4", "#8EC9FF", "#FFE59A"], last = 0;
  document.addEventListener("pointermove", function (e) {
    var now = Date.now();
    if (now - last < 90 || !e.target.closest(".btn, .cards article, .love, nav a")) { return; }
    last = now;
    var s = document.createElement("span");
    s.className = "spark";
    s.textContent = glyphs[now % 3];
    s.style.left = e.clientX + "px";
    s.style.top = e.clientY + "px";
    s.style.color = colors[now % 4];
    s.setAttribute("aria-hidden", "true");
    document.body.appendChild(s);
    setTimeout(function () { s.remove(); }, 700);
  });
})();
