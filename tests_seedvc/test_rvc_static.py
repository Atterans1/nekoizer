"""Static checks for the RVC UI slice: ids, routes and banned markup in the static files."""

import re
import unittest
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "ui" / "static"

PRE_EXISTING_IDS = [
    "st-pipeline", "st-input", "st-output", "st-latency", "st-underruns", "st-balancer",
    "slots", "sliders", "presets", "msg", "url-slot", "url-input", "url-go", "library-open",
    "monitor_enabled", "auto_balancer", "emo-auto", "emo-manual", "h-refs", "h-rack", "h-presets",
    "controls",
]

PREFIX_IDS = [
    "rvc-open-", "rvc-panel-", "rvc-file-", "rvc-link-", "rvc-link-go-",
    "rvc-use-", "rvc-use-go-", "rvc-del-", "ctl-note-",
]

ROUTES = [
    "/api/rvc/models", "/api/rvc/models/", "/rvc/import", "/rvc/link", "/rvc/use",
    "/api/open_voice_library",
]

SCRIPT_RE = re.compile(r"<script\b([^>]*)>(.*?)</script>", re.S | re.I)


def ReadText(name):
    return (STATIC / name).read_text(encoding="utf-8")


class RvcStaticTest(unittest.TestCase):
    def setUp(self):
        self.html = ReadText("index.html")
        self.js = ReadText("app.js")
        self.css = ReadText("style.css")

    def test_prefix_ids_are_literals_in_app_js(self):
        for prefix in PREFIX_IDS:
            with self.subTest(prefix=prefix):
                self.assertTrue(('"%s"' % prefix) in self.js or ("'%s'" % prefix) in self.js)

    def test_weights_and_engine_ids_exist(self):
        for ident in ("rvc-weights", "st-engine"):
            with self.subTest(ident=ident):
                self.assertIn(ident, self.html + self.js)

    def test_pre_existing_ids_still_present(self):
        for ident in PRE_EXISTING_IDS:
            with self.subTest(ident=ident):
                self.assertIn('id="%s"' % ident, self.html)

    def test_route_strings_used_by_app_js(self):
        for route in ROUTES:
            with self.subTest(route=route):
                self.assertIn(route, self.js)

    def test_html_has_no_external_host_or_inline_style(self):
        self.assertNotIn("http://", self.html)
        self.assertNotIn("https://", self.html)
        self.assertIsNone(re.search(r"\sstyle\s*=", self.html))

    def test_html_scripts_are_external_only(self):
        for attrs, body in SCRIPT_RE.findall(self.html):
            with self.subTest(attrs=attrs):
                self.assertIn("src=", attrs)
                self.assertEqual(body.strip(), "")

    def test_html_has_honest_web_voice_text(self):
        self.assertIn("Only use voices you have the right to use.", self.html)
        self.assertNotIn("which Nekoizer cannot run", self.html)

    def test_js_and_css_have_no_external_host_or_inner_html(self):
        for source in (self.js, self.css):
            self.assertNotIn("http://", source)
            self.assertNotIn("https://", source)
            self.assertNotIn("innerHTML", source)


if __name__ == "__main__":
    unittest.main()
