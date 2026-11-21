"""Every weights file read by the new RVC code goes through the weights_only loader: one torch.load call, no other deserializer, no alias that hides either."""
import ast
import glob
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAFE_LOADER = os.path.join(ROOT, "engine", "rvc_checkpoint.py")
UNSAFE_MODULES = {"pickle", "cPickle", "_pickle", "dill", "cloudpickle", "marshal", "shelve", "joblib"}
UNSAFE_CALLS = {"torch.jit.load", "torch.hub.load", "torch.serialization.load", "np.load", "numpy.load"}
UNSAFE_TORCH_NAMES = {"load", "jit", "hub", "serialization"}


def NewCodeFiles():
    files = [os.path.join(ROOT, name) for name in ("net_guard.py", "pipeline.py", "ui/server.py", "ui/fake_pipeline.py",
                                                  "engine/engine_host.py", "engine/voice_engine.py")]
    files += glob.glob(os.path.join(ROOT, "engine", "rvc_*.py"))
    files += glob.glob(os.path.join(ROOT, "third_party", "rvc", "**", "*.py"), recursive=True)
    return files


def Parse(path):
    with open(path, encoding="utf-8") as handle:
        return ast.parse(handle.read(), path)


def Dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


class NoPlainLoadTests(unittest.TestCase):
    def test_the_scan_covers_the_vendored_folder_and_every_new_module(self):
        files = NewCodeFiles()
        for path in files:
            self.assertTrue(os.path.isfile(path), path)
        names = {os.path.relpath(p, ROOT).replace(os.sep, "/") for p in files}
        self.assertIn("third_party/rvc/rmvpe.py", names)
        self.assertIn("engine/rvc_checkpoint.py", names)
        self.assertIn("engine/engine_host.py", names)

    def test_every_torch_load_call_passes_weights_only_true_and_there_is_exactly_one(self):
        calls = []
        for path in NewCodeFiles():
            for node in ast.walk(Parse(path)):
                if isinstance(node, ast.Call) and Dotted(node.func) == "torch.load":
                    flags = {k.arg: getattr(k.value, "value", None) for k in node.keywords}
                    self.assertIs(flags.get("weights_only"), True, f"{path}:{node.lineno} torch.load without weights_only=True")
                    calls.append(path)
        self.assertEqual(calls, [SAFE_LOADER])

    def test_no_other_deserializer_and_no_alias_that_hides_torch_load(self):
        for path in NewCodeFiles():
            for node in ast.walk(Parse(path)):
                where = f"{path}:{getattr(node, 'lineno', 0)}"
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn(alias.name.split(".")[0], UNSAFE_MODULES, f"{where} imports a general deserializer")
                        self.assertFalse(alias.name == "torch" and alias.asname, f"{where} aliases torch, which hides torch.load from this scan")
                elif isinstance(node, ast.ImportFrom):
                    root = (node.module or "").split(".")[0]
                    self.assertNotIn(root, UNSAFE_MODULES, f"{where} imports a general deserializer")
                    if node.module in ("torch", "torch.serialization", "torch.jit", "torch.hub"):
                        names = {alias.name for alias in node.names}
                        self.assertFalse(names & UNSAFE_TORCH_NAMES, f"{where} imports a torch loader by name")
                elif isinstance(node, ast.Call):
                    target = Dotted(node.func)
                    self.assertNotIn(target, UNSAFE_CALLS, f"{where} calls {target}")
                    self.assertNotEqual(getattr(node.func, "attr", None), "from_pretrained", f"{where} calls from_pretrained")


if __name__ == "__main__":
    unittest.main()
