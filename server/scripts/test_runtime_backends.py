"""Backend selection, provider isolation, config integrity and unchanged HTTP contract."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from feiyu_runtime.common import (apply_backend_env, backend_name, cpu_onnx_sessions, device_type, load,
                                 prepare, save, seed_all, sha, validate_profile, verify_package)
from feiyu_singing import pipeline
from feiyu_voice_service.registry import Registry
from feiyu_voice_service.worker import execute
sys.path.insert(0, str(ROOT / "scripts"))
import register_runtime


def profile(engine="tts", backend="hip"):
    return dict(schema="feiyu-runtime-v1", engine=engine, backend=backend,
        distribution="Ubuntu-24.04", python="/test/environment/bin/python", repo="/test/source",
        commit="test-commit", torch_version="test-version", torchaudio_version="test-version",
        source_sha256={"entry.py": "test-digest"}, precision="float32", f0_device="cpu",
        onnx_provider="CPUExecutionProvider")


def torch_stub(backend="hip", available=True):
    # Only the build matching `backend` has its accelerator namespace usable.
    def accelerator(name):
        return SimpleNamespace(is_available=lambda: available and backend == name,
                               get_device_name=lambda n: "test GPU", manual_seed_all=Mock())
    return SimpleNamespace(__version__="test-version",
        version=SimpleNamespace(hip="7.2" if backend == "hip" else None,
                                cuda="12.8" if backend == "cuda" else None,
                                xpu="20260100" if backend == "xpu" else None),
        cuda=accelerator("hip" if backend == "hip" else "cuda"), xpu=accelerator("xpu"),
        manual_seed=Mock(),
        backends=SimpleNamespace(cuda=SimpleNamespace(enable_flash_sdp=Mock(),
            enable_mem_efficient_sdp=Mock(), enable_math_sdp=Mock())), set_num_threads=Mock())


class RuntimeTests(unittest.TestCase):
    def test_register_keeps_virtualenv_interpreter_symlink(self):
        # Resolving env/bin/python changes sys.prefix and loses installed packages.
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "runtime.json"
            virtual = Path(tmp) / "venv/bin/python"
            argv = ["register_runtime.py", "--backend", "cuda", "--engine", "tts", "--repo", tmp,
                    "--python", str(virtual), "--output", str(output)]
            with patch.object(sys, "argv", argv), patch.object(register_runtime.subprocess, "check_output",
                    return_value=register_runtime.SOURCES["tts"][1]), patch.object(register_runtime, "run"), \
                    patch.object(register_runtime, "build_profile", return_value=profile()) as build:
                register_runtime.main()
            self.assertEqual(build.call_args.args[3], virtual.absolute())

    def test_rocm_preset_has_no_nvidia_only_inference_requirements(self):
        preset = load(ROOT / "runtime_profiles/rocm.json")
        self.assertEqual(preset["backend"], "hip")
        for engine, stack in preset["engines"].items():
            self.assertIn("+rocm", stack["torch"])
            requirements = (ROOT / "runtime_profiles" / ("requirements-" + engine + ".txt")).read_text()
            packages = [line.lower() for line in requirements.splitlines() if line and not line.startswith("#")]
            self.assertFalse(any(name.startswith(("nvidia-", "tensorrt", "deepspeed", "sageattention", "onnxruntime-gpu")) for name in packages))
        self.assertEqual(preset["engines"]["rvc"]["f0_device"], "cpu")

    def test_hip_uses_cuda_device_name(self):
        result = prepare(torch_stub(), profile())
        self.assertEqual(result["backend"], "hip")
        self.assertEqual(result["device"], "cuda:0")

    def test_xpu_preset_is_fp32_and_installs_only_xpu_wheels(self):
        preset = load(ROOT / "runtime_profiles/xpu.json")
        self.assertEqual(preset["backend"], "xpu")
        self.assertTrue(preset["experimental"])
        self.assertTrue(preset["torch_index"].endswith("/xpu"))
        for stack in preset["engines"].values():
            self.assertTrue(stack["torch"].endswith("+xpu"))
            self.assertEqual(stack["precision"], "float32")

    def test_xpu_uses_xpu_device_and_leaves_cuda_sdp_alone(self):
        torch = torch_stub("xpu")
        result = prepare(torch, profile(backend="xpu"))
        self.assertEqual((result["backend"], result["device"], result["gpu"]), ("xpu", "xpu:0", "test GPU"))
        self.assertEqual(result["xpu"], "20260100")
        torch.backends.cuda.enable_flash_sdp.assert_not_called()

    def test_xpu_profile_refuses_other_wheels_and_fp16(self):
        for torch in (torch_stub("cuda"), torch_stub("hip"), torch_stub("xpu", False)):
            with self.assertRaises(RuntimeError):
                prepare(torch, profile(backend="xpu"))
        with self.assertRaises(RuntimeError):
            prepare(torch_stub("xpu"), profile(backend="cuda"))
        p = profile("svs", "xpu")
        p["precision"] = "float16"
        with self.assertRaises(ValueError):
            validate_profile(p)

    def test_backend_helpers(self):
        self.assertEqual([backend_name(torch_stub(b)) for b in ("hip", "cuda", "xpu")], ["hip", "cuda", "xpu"])
        self.assertEqual([device_type(b) for b in ("cpu", "cuda", "hip", "xpu")], ["cpu", "cuda", "cuda", "xpu"])
        torch = torch_stub("xpu")
        seed_all(torch, "xpu", 7)
        torch.manual_seed.assert_called_once_with(7)
        torch.xpu.manual_seed_all.assert_called_once_with(7)
        torch.cuda.manual_seed_all.assert_not_called()

    def test_xpu_sets_fallback_environment_without_claiming_operator_coverage(self):
        with patch.dict("os.environ", {}, clear=False):
            sys.modules["os"].environ.pop("PYTORCH_ENABLE_XPU_FALLBACK", None)
            apply_backend_env(profile(backend="cuda"))
            self.assertNotIn("PYTORCH_ENABLE_XPU_FALLBACK", sys.modules["os"].environ)
            apply_backend_env(profile(backend="xpu"))
            self.assertEqual(sys.modules["os"].environ["PYTORCH_ENABLE_XPU_FALLBACK"], "1")

    def test_native_windows_host_skips_wsl_wrapping_and_path_mapping(self):
        native = dict(profile(backend="xpu"), host="windows", python="C:/env/python.exe")
        with patch.object(pipeline, "os", SimpleNamespace(name="nt")):
            command = pipeline.runtime_command(native, ROOT / "feiyu_runtime/worker.py", "--kind", "tts")
            self.assertEqual(command[0], "C:/env/python.exe")
            self.assertNotIn("wsl.exe", command)
            self.assertEqual(pipeline.linux_path(ROOT, native), str(ROOT.resolve()))
            with patch.object(pipeline, "linux_path", return_value="/mnt/c/repo/worker.py"):
                wsl = pipeline.runtime_command(profile(), ROOT / "feiyu_runtime/worker.py")
            self.assertEqual(wsl[:3], ["wsl.exe", "-d", "Ubuntu-24.04"])
            self.assertIn("/mnt/c/repo/worker.py", wsl)

    def test_native_linux_xpu_runs_python_directly(self):
        native = dict(profile(backend="xpu"), host="wsl")
        with patch.object(pipeline, "os", SimpleNamespace(name="posix")):
            command = pipeline.runtime_command(native, ROOT / "feiyu_runtime/worker.py", "--kind", "tts")
        self.assertEqual(command[0], "/test/environment/bin/python")
        self.assertNotIn("wsl.exe", command)
        self.assertEqual(command[2], str((ROOT / "feiyu_runtime/worker.py").resolve()))

    def test_windows_paths_are_stateless_across_runtime_switches(self):
        from pathlib import PureWindowsPath
        class WindowsPath(PureWindowsPath):
            def resolve(self):
                return self
        native = dict(profile(backend="xpu"), host="windows", python="C:/env/python.exe")
        wsl = profile(backend="cuda")
        with patch.object(pipeline, "os", SimpleNamespace(name="nt")), patch.object(pipeline, "Path", WindowsPath):
            for first, second in ((native, wsl), (wsl, native)):
                pipeline.runtime_command(first, "C:/repo/worker.py")
                argument = pipeline.linux_path("C:/jobs/vocal.wav", second)
                command = pipeline.runtime_command(second, "C:/repo/worker.py", "--input", argument)
                if second is native:
                    self.assertEqual(command[0], "C:/env/python.exe")
                    self.assertEqual(command[-1], "C:\\jobs\\vocal.wav")
                else:
                    self.assertEqual(command[0], "wsl.exe")
                    self.assertEqual(command[-1], "/mnt/c/jobs/vocal.wav")
            # Legacy calls remain WSL mappings, even after a native command.
            pipeline.runtime_command(native, "C:/repo/worker.py")
            self.assertEqual(pipeline.linux_path("C:/jobs/vocal.wav"), "/mnt/c/jobs/vocal.wav")

    def test_singing_serializes_each_stage_for_its_own_host(self):
        """Exercise real orchestration for all native/WSL SVS->RVC combinations."""
        from feiyu_singing import score as score_module
        import itertools
        for svs_host, rvc_host in itertools.product(("windows", "wsl"), repeat=2):
            with self.subTest(svs=svs_host, rvc=rvc_host), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                package = root / "package"
                package.mkdir()
                save(package / "manifest.json", dict(id="test-rvc", schema="feiyu-rvc-release-v1"))
                for name in ("score.abc", "weight.pt", "prompt.wav", "prompt.json"):
                    (root / name).write_bytes(b"test fixture")
                save(root / "phones.json", ["a"])
                svs = dict(profile("svs", "xpu"), host=svs_host)
                rvc = dict(profile("rvc", "xpu"), host=rvc_host)
                save(root / "svs.json", svs)
                save(root / "rvc.json", rvc)
                config = dict(rvc_package=str(package), rvc_package_id="test-rvc", svs_runtime=svs,
                    rvc_runtime=rvc, svs_runtime_profile=str(root / "svs.json"), rvc_runtime_profile=str(root / "rvc.json"),
                    phoneset=dict(path=str(root / "phones.json"), sha256=sha(root / "phones.json")),
                    svs_weight=dict(path=str(root / "weight.pt"), sha256=sha(root / "weight.pt")),
                    prompt=dict(id="test", annotation_status="test", **{
                        role: dict(path=str(root / name), sha256=sha(root / name))
                        for role, name in (("audio", "prompt.wav"), ("metadata", "prompt.json"))}))
                save(root / "config.json", config)
                output = root / "output"
                compiled = dict(events=[dict(phoneme="a")], duration_seconds=1, metadata=[], frames=50)
                commands = []

                def mapped(path, runtime=None):
                    self.assertIsNotNone(runtime, "Every stage-bound path needs its target runtime")
                    return runtime["host"] + ":" + str(path)

                def run(command, log):
                    commands.append(command)
                    if len(commands) == 1:
                        (output / "svs/generated.wav").write_bytes(b"svs audio fixture")
                        save(output / "svs/result.json", dict(status="success", output_sha256=sha(output / "svs/generated.wav")))
                    else:
                        (output / "rvc").mkdir()
                        (output / "rvc/rvc_native.wav").write_bytes(b"rvc audio fixture")
                        save(output / "rvc/cover_result.json", dict(status="success", package="test-rvc",
                            native_sha256=sha(output / "rvc/rvc_native.wav"), breath_attenuation_applied=False))

                def finalize(source, dest, seconds):
                    dest.write_bytes(source.read_bytes())
                    return dict(sample_rate=40000)

                with patch.object(score_module, "read_score", return_value=([], 1, {})), \
                     patch.object(score_module, "compile_score", return_value=compiled), \
                     patch.object(pipeline, "linux_path", side_effect=mapped), \
                     patch.object(pipeline, "run_logged", side_effect=run), \
                     patch.object(pipeline, "finalize_audio", side_effect=finalize), patch.object(pipeline, "make_preview"):
                    pipeline.synthesize(root / "score.abc", lyrics="a", output_dir=output, config=root / "config.json")
                job = load(output / "svs/job.json")
                for item in [job["weight"], job["phoneset"], job["target"], *job["prompt"].values()]:
                    self.assertTrue(item["path"].startswith(svs_host + ":"))
                self.assertTrue(job["output_dir"].startswith(svs_host + ":"))
                self.assertTrue(commands[0][-1].startswith(svs_host + ":"))
                command = commands[1]
                for option in ("--package", "--profile", "--input", "--output-dir"):
                    self.assertTrue(command[command.index(option) + 1].startswith(rvc_host + ":"))

    def test_unknown_host_rejected(self):
        with self.assertRaises(ValueError):
            validate_profile(dict(profile(backend="xpu"), host="macos"))

    def test_wrong_wheel_and_missing_gpu_fail(self):
        for torch in (torch_stub("cuda"), torch_stub("hip", False)):
            with self.assertRaises(RuntimeError):
                prepare(torch, profile())

    def test_torch_version_drift_fails(self):
        p = profile()
        p["torch_version"] = "another-version"
        with self.assertRaises(ValueError):
            prepare(torch_stub(), p)

    def test_cuda_profile_preserves_backend_defaults(self):
        torch = torch_stub("cuda")
        self.assertEqual(prepare(torch, profile(backend="cuda"))["backend"], "cuda")
        torch.backends.cuda.enable_flash_sdp.assert_not_called()

    def test_incomplete_or_unsafe_profiles_rejected(self):
        for changes in ({"source_sha256": {}}, {"precision": "float16"},
                        {"onnx_provider": "CUDAExecutionProvider"}, {"f0_device": "unknown"},
                        {"backend": "rocm"}, {"engine": "unknown"}):
            p = profile()
            p.update(changes)
            with self.assertRaises(ValueError):
                validate_profile(p)

    def test_onnx_cpu_override_restored_on_error(self):
        factory = Mock(return_value=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]))
        ort = SimpleNamespace(InferenceSession=factory)
        sessions = []
        with self.assertRaisesRegex(RuntimeError, "after initialization"):
            with cpu_onnx_sessions(ort, sessions):
                ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
                raise RuntimeError("after initialization")
        self.assertIs(ort.InferenceSession, factory)
        self.assertEqual(factory.call_args.kwargs["providers"], ["CPUExecutionProvider"])
        self.assertEqual(len(sessions), 1)

    def test_model_asset_tampering_and_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "model.pt").write_bytes(b"original")
            save(folder / "manifest.json", {})
            save(folder / "files_sha256.json", {"model.pt": dict(sha256=sha(folder / "model.pt"))})
            verify_package(folder)
            (folder / "model.pt").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_package(folder)
            save(folder / "files_sha256.json", {"../outside.pt": dict(sha256="wrong")})
            with self.assertRaises(ValueError):
                verify_package(folder)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = dict(schema="feiyu-voice-service-v1", data_dir=str(self.root / "data"), tts={}, rvc={})
        for kind, schema, script in [("tts", "feiyu-voice-local-release-v1", "infer.py"),
                                     ("rvc", "feiyu-rvc-release-v1", "run_rvc.py")]:
            package = self.root / kind
            package.mkdir()
            save(package / "manifest.json", dict(schema=schema, id="FeiyuVoice-RVC-v1", version="test", styles={"cute": {}}))
            save(package / "files_sha256.json", {})
            (package / script).write_text("# fixture; never executed")
            p = self.root / (kind + ".local.json")
            save(p, profile(kind))
            self.config[kind]["feiyu"] = dict(engine="cosyvoice3_release" if kind == "tts" else "feiyu_rvc_release",
                package=str(package), runtime_profile=str(p))
        self.path = self.root / "service.json"
        save(self.path, self.config)

    def tearDown(self):
        self.temp.cleanup()

    def test_profiles_replace_package_local_runtime_without_editing_assets(self):
        registry = Registry(self.path)
        self.assertTrue(registry.public()["tts"][0]["available"])
        self.assertTrue(registry.public()["rvc"][0]["available"])
        self.assertEqual(registry.public()["tts"][0]["runtime_backend"], "hip")
        self.assertNotIn(str(self.root), str(registry.public()))
        self.assertIn("not yet checked", registry.public()["tts"][0]["runtime_verification"])
        self.assertFalse((self.root / "tts/runtime.local.json").exists())

    def test_engine_mismatch_is_unavailable(self):
        save(self.root / "tts.local.json", profile("rvc"))
        self.assertFalse(Registry(self.path).public()["tts"][0]["available"])

    def test_changed_profile_after_submit_fails(self):
        entry = Registry(self.path).select("speech", "feiyu")
        save(self.root / "tts.local.json", profile("tts", "cuda"))
        with self.assertRaisesRegex(ValueError, "changed after submission"):
            execute(dict(kind="speech", backend=entry, request={}, job_directory=str(self.root)))

    def test_changed_adapter_after_submit_fails(self):
        entry = Registry(self.path).select("speech", "feiyu")
        entry["adapter_sha256"] = "stale"
        with self.assertRaisesRegex(ValueError, "adapter changed"):
            execute(dict(kind="speech", backend=entry, request={}, job_directory=str(self.root)))

    def test_worker_launches_selected_environment(self):
        for kind in ("speech", "conversion"):
            folder = self.root / kind
            folder.mkdir()
            entry = Registry(self.path).select(kind, "feiyu")
            job = dict(kind=kind, backend=entry, request=dict(text="你好", style="cute"),
                       audio_path=str(self.root / "vocal.wav"), job_directory=str(folder))
            with patch("feiyu_voice_service.worker.run_logged", side_effect=RuntimeError("stop before inference")) as run:
                with self.assertRaisesRegex(RuntimeError, "stop before inference"):
                    execute(job)
            command = run.call_args.args[0]
            self.assertIn("/test/environment/bin/python", command)
            self.assertIn("--profile", command)
            self.assertIn("--package", command)
            self.assertIn("tts" if kind == "speech" else "rvc", command)

    def test_unchanged_http_requests_with_managed_profiles(self):
        from fastapi.testclient import TestClient
        from feiyu_voice_service.app import create_app
        def fake_runner(folder):
            import wave
            with wave.open(str(folder / "audio.wav"), "wb") as audio:
                audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                audio.writeframes(b"\0\1" * 2400)
            return dict(sha256=sha(folder / "audio.wav"), audio=dict(sample_rate=24000, seconds=.1))
        with TestClient(create_app(self.path, api_key="test-only-not-a-production-key", runner=fake_runner),
                        headers={"Authorization": "Bearer test-only-not-a-production-key"}) as client:
            self.assertEqual(client.post("/v1/speech", json=dict(text="你好")).status_code, 202)
            asset = client.post("/v1/assets", files={"file": ("vocal.wav", b"fixture")}).json()
            self.assertEqual(client.post("/v1/voice-conversion", json=dict(audio_id=asset["id"])).status_code, 202)


if __name__ == "__main__":
    unittest.main()
