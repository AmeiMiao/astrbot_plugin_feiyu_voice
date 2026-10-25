"""CPU-only API contract checks with temporary package fixtures and a fake runner."""
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import time
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from feiyu_singing.pipeline import save, sha
from feiyu_voice_service.app import create_app


def wav_bytes():
    stream = BytesIO()
    with wave.open(stream, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(24000)
        out.writeframes(b"\x00\x01" * 2400)
    return stream.getvalue()


def runner(folder):
    (folder / "audio.wav").write_bytes(wav_bytes())
    return dict(sha256=sha(folder / "audio.wav"), audio=dict(sample_rate=24000, seconds=.1))


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for kind, manifest, script in [
            ("tts", dict(schema="feiyu-voice-local-release-v1", version="test", styles=dict(cute={})), "infer.py"),
            ("rvc", dict(schema="feiyu-rvc-release-v1", version="1.0.0", id="FeiyuVoice-RVC-v1"), "run_rvc.py")
        ]:
            package = self.root / kind
            package.mkdir()
            save(package / "manifest.json", manifest)
            save(package / "runtime.local.json", {})
            save(package / "files_sha256.json", {})
            (package / script).write_text("# Test fixture; inference is mocked")
        config = dict(schema="feiyu-voice-service-v1", data_dir=str(self.root / "data"), max_queue=2,
            tts=dict(feiyu=dict(engine="cosyvoice3_release", package=str(self.root / "tts"))),
            rvc=dict(feiyu=dict(engine="feiyu_rvc_release", package=str(self.root / "rvc"))), singing=None)
        self.config = self.root / "service.json"
        save(self.config, config)

    def tearDown(self):
        self.temp.cleanup()

    def client(self):
        return TestClient(create_app(self.config, api_key="test-key-not-for-production", runner=runner),
                          headers={"Authorization":"Bearer test-key-not-for-production"})

    def test_auth_and_optional_capabilities(self):
        with self.client() as client:
            self.assertEqual(client.get("/v1/capabilities", headers={"Authorization":""}).status_code, 401)
            result = client.get("/v1/capabilities").json()
            self.assertTrue(result["tts"][0]["available"])
            self.assertEqual(result["rvc"][0]["package_id"], "FeiyuVoice-RVC-v1")
            self.assertFalse(result["singing"]["available"])

    def test_speech_and_rvc_jobs_download(self):
        with self.client() as client:
            asset = client.post("/v1/assets", files={"file":("vocal.wav", wav_bytes())}).json()["id"]
            for route, payload in [("speech", dict(text="你好")), ("voice-conversion", dict(audio_id=asset))]:
                submitted = client.post("/v1/" + route, json=payload)
                self.assertEqual(submitted.status_code, 202)
                job_id = submitted.json()["id"]
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    result = client.get("/v1/jobs/" + job_id).json()
                    if result["status"] == "success":
                        break
                    time.sleep(.02)
                self.assertEqual(result["status"], "success")
                self.assertEqual(client.get(result["audio_url"]).content, wav_bytes())


if __name__ == "__main__":
    unittest.main()
