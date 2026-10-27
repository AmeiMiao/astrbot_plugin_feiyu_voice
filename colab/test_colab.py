"""Offline contracts, not a substitute for real Colab GPU acceptance."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import feiyu_colab as fv


class NotebookTests(unittest.TestCase):
    def test_notebooks_clean_valid_and_compilable(self):
        import nbformat
        for path in fv.HERE.glob("*.ipynb"):
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            source = path.read_text(encoding="utf-8")
            for forbidden in ("C:\\", "/home/", "hf_", "ngrok", "drive.mount", "launch(share"):
                self.assertNotIn(forbidden, source)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    self.assertEqual(cell.outputs, [])
                    self.assertIsNone(cell.execution_count)
                    compile(cell.source, str(path), "exec")
            bootstrap = next(c.source for c in notebook.cells if c.get("id") == "bootstrap")
            for name in ("feiyu_colab.py", "releases.json"):
                self.assertIn(fv.sha(fv.HERE / name), bootstrap)

    def test_release_pins(self):
        for name in ("feiyu_colab.py", "releases.json"):
            self.assertNotIn(b"\r\n", (fv.HERE / name).read_bytes())
        for engine in ("rvc", "tts"):
            release = fv.LOCK[engine]
            for field, length in (("revision", 40), ("manifest_sha256", 64), ("checksums_sha256", 64)):
                self.assertEqual(len(release[field]), length)
                int(release[field], 16)
        self.assertEqual(len(fv.LOCK["source_commit"]), 40)

    def test_bootstrap_downloads_verified_helpers_and_reuses_them(self):
        notebook = json.loads((fv.HERE / "FeiyuVoice_RVC.ipynb").read_text(encoding="utf-8"))
        code = next(c["source"] for c in notebook["cells"] if c["id"] == "bootstrap")
        with tempfile.TemporaryDirectory() as directory:
            code = code.replace('pathlib.Path("/content/feiyuvoice")', f'pathlib.Path({directory!r})')
            calls = []

            def fetch(url, **kwargs):
                self.assertTrue(url.startswith("https://raw.githubusercontent.com/Esmerubphire/astrbot_plugin_feiyu_voice/main/colab/"))
                calls.append(url)
                return io.BytesIO((fv.HERE / url.rsplit("/", 1)[1]).read_bytes())

            # Keep the real module import/verification; avoid actually checking host GPU.
            code = code.replace("fv.check_gpu()", "assert callable(fv.check_gpu)")
            with patch("urllib.request.urlopen", side_effect=fetch):
                exec(compile(code, "bootstrap", "exec"), {})
                exec(compile(code, "bootstrap", "exec"), {})
                self.assertEqual(len(calls), 2)

    def test_parameter_boundaries(self):
        fv.validate_parameters(0, 0.3, 0.33)
        fv.validate_parameters(-24, 0, 0)
        fv.validate_parameters(24, 1, 0.5)
        for params in ((25, .3, .33), (.5, .3, .33), (True, .3, .33),
                       (0, float("nan"), .33), (0, 1.01, .33), (0, .3, .51)):
            with self.assertRaises(ValueError):
                fv.validate_parameters(*params)

    def test_package_traversal_and_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                fv.safe_path(directory, "../escape")
            with patch.object(fv.shutil, "disk_usage", return_value=types.SimpleNamespace(free=1)):
                with self.assertRaisesRegex(RuntimeError, "磁盘空间不足"):
                    fv.need_disk(directory, 10)

    def test_no_gpu_fails_early(self):
        with patch.dict(sys.modules, {"torch": None}), patch.object(
            fv.subprocess, "check_output", side_effect=FileNotFoundError
        ), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "没有可用 NVIDIA GPU"):
                fv.check_gpu()

    def test_download_reuses_good_files_repairs_corruption_without_token(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "tts/package"
            package.mkdir(parents=True)
            manifest = b'{"styles": {}}'
            data = b"public weight fixture"
            digest = lambda b: hashlib.sha256(b).hexdigest()
            contents = {"manifest.json": manifest, "models/test.pt": data}
            checksums = json.dumps({n: dict(bytes=len(b), sha256=digest(b)) for n, b in contents.items()}).encode()
            contents["files_sha256.json"] = checksums
            lock = dict(tts=dict(repo="test/public", revision="a" * 40,
                                checksums_sha256=digest(checksums), manifest_sha256=digest(manifest)))
            calls = []

            def download(repo, name, **kwargs):
                self.assertIs(kwargs["token"], False)
                self.assertEqual(kwargs["revision"], "a" * 40)
                calls.append((name, kwargs["force_download"]))
                target = package / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(contents[name])
                return target

            with patch.dict(sys.modules, {"huggingface_hub": types.SimpleNamespace(hf_hub_download=download)}), \
                 patch.object(fv, "LOCK", lock), patch.object(fv, "need_disk"), contextlib.redirect_stdout(io.StringIO()):
                fv.download("tts", directory)
                self.assertEqual(len(calls), 3)
                fv.download("tts", directory)
                self.assertEqual(len(calls), 3)
                (package / "models/test.pt").write_bytes(b"broken")
                fv.download("tts", directory)
                self.assertEqual(calls[-1], ("models/test.pt", True))

    def test_audio_checks_before_conversion_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.wav"
            source.write_bytes(b"original")
            dest = Path(directory) / "decoded.wav"
            with patch.object(fv.subprocess, "check_output", return_value=json.dumps(
                dict(streams=[dict(codec_type="audio")], format=dict(duration="4.0"))
            )), patch.object(fv, "run") as run:
                self.assertEqual(fv.prepare_audio(source, dest), 4)
                self.assertIn("-n", run.call_args.args[0])
                self.assertEqual(source.read_bytes(), b"original")
            for duration in ("nan", "0.1", "601"):
                with patch.object(fv.subprocess, "check_output", return_value=json.dumps(
                    dict(streams=[{}], format=dict(duration=duration))
                )), self.assertRaises(ValueError):
                    fv.prepare_audio(source, dest)
            dest.write_bytes(b"existing")
            with patch.object(fv.subprocess, "check_output", return_value=json.dumps(
                dict(streams=[{}], format=dict(duration="4"))
            )), self.assertRaisesRegex(ValueError, "已存在"):
                fv.prepare_audio(source, dest)

    def test_ready_setup_repeat_does_not_reinstall_or_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "rvc"
            state.mkdir()
            fv.write(state / "ready.json", dict(lock_sha256=fv.sha(fv.HERE / "releases.json")))
            fv.write(state / "runtime.local.json", dict(python="/test/python"))
            fv.write(state / "diagnostic.json", dict(status="success"))
            with patch.object(fv.sys, "platform", "linux"), patch.object(fv.platform, "machine", return_value="x86_64"), \
                 patch.object(fv, "check_gpu"), patch.object(fv, "run") as run, \
                 patch.object(fv, "clone_pinned") as clone, contextlib.redirect_stdout(io.StringIO()):
                fv.setup("rvc", root)
                fv.setup("rvc", root)
                self.assertEqual(run.call_count, 2)  # actual diagnostic on every initialization
                clone.assert_not_called()

    def test_tts_rejects_unknown_style_and_empty_text(self):
        for request in (dict(style="other", text="hello"), dict(style="cute", text="  ")):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "request.json"
                fv.write(path, request)
                with self.assertRaises(ValueError):
                    fv.infer_tts(directory, path)

    def test_tts_uses_published_builtin_frontend_without_unpinned_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "requirements.txt"
            source.write_text("numpy==1.26.4\nwetext==0.0.4\ntransformers==4.51.3\n")
            output = Path(directory) / "colab.txt"
            fv.inference_requirements("tts", source, output)
            self.assertEqual(output.read_text(), "numpy==1.26.4\ntransformers==4.51.3\n")
            path = Path(directory) / "request.json"
            fv.write(path, dict(style="cute", text="hello"))
            with patch.object(fv.importlib.util, "find_spec", return_value=object()):
                with self.assertRaisesRegex(RuntimeError, "WeText"):
                    fv.infer_tts(directory, path)

    def test_each_generation_uses_unique_output_and_selected_style(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(fv, "worker") as worker:
            first = fv.submit("tts", directory, text="hello", style="gentle")
            second = fv.submit("tts", directory, text="hello", style="cute")
            self.assertNotEqual(first, second)
            requests = [fv.read(c.args[3]) for c in worker.call_args_list]
            self.assertEqual([r["style"] for r in requests], ["gentle", "cute"])


if __name__ == "__main__":
    unittest.main()
