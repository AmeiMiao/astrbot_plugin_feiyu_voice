"""A durable local queue with a single inference worker and bounded submissions."""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from feiyu_singing.pipeline import ROOT, load, save, sha

ID = re.compile(r"^[0-9a-f]{32}$")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def persist_status(folder, record):
    temporary = folder / ".status.json.tmp"
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(folder / "status.json")


class JobManager:
    def __init__(self, data_dir, max_queue=16, runner=None):
        self.root = Path(data_dir).resolve()
        self.jobs = self.root / "jobs"
        self.assets = self.root / "assets"
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.assets.mkdir(exist_ok=True)
        self.queue = queue.Queue(maxsize=max_queue)
        self.lock = threading.RLock()
        self.records = {}
        self.active = None
        self.process = None
        self.runner = runner or self.run_process
        self.stopping = threading.Event()
        for path in self.jobs.glob("*/status.json"):
            row = load(path)
            self.records[row["id"]] = row

    def start(self):
        # Called only after the application acquired its exclusive storage lock.
        for row in self.records.values():
            if row["status"] in ("queued", "running"):
                row.update(
                    status="interrupted",
                    finished_at=timestamp(),
                    error="Service restarted before this job completed",
                )
                persist_status(self.jobs / row["id"], row)
        self.thread = threading.Thread(
            target=self.work, name="feiyu-inference", daemon=True
        )
        self.thread.start()

    def stop(self):
        self.stopping.set()
        with self.lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()
        self.thread.join(timeout=5)

    def asset(self, asset_id, suffixes=None):
        if not ID.fullmatch(asset_id):
            raise ValueError("Invalid asset ID")
        meta = load(self.assets / (asset_id + ".json"))
        path = (self.assets / meta["stored_name"]).resolve()
        if (
            path.parent != self.assets
            or not path.is_file()
            or sha(path) != meta["sha256"]
        ):
            raise ValueError("Uploaded asset is missing or changed")
        if suffixes and path.suffix.lower() not in suffixes:
            raise ValueError("Wrong asset type for this request")
        return path

    def submit(self, kind, request, backend):
        with self.lock:
            if self.stopping.is_set() or self.queue.full():
                raise queue.Full("Inference queue is full")
            job_id = uuid.uuid4().hex
            folder = self.jobs / job_id
            folder.mkdir()
            payload = dict(
                kind=kind, request=request, backend=backend, job_directory=str(folder)
            )
            for key, suffixes in (
                ("audio_id", {".wav", ".flac"}),
                ("score_id", {".mid", ".midi", ".abc"}),
                ("lyrics_id", {".txt"}),
                ("pinyin_id", {".txt"}),
                ("regions_id", {".json"}),
            ):
                if request.get(key):
                    source = self.asset(request[key], suffixes)
                    # The worker receives only server-resolved paths; no arbitrary client filesystem access.
                    payload[key.removesuffix("_id") + "_path"] = str(source)
            save(folder / "request.json", payload)
            record = dict(
                id=job_id,
                kind=kind,
                status="queued",
                created_at=timestamp(),
                voice=request["voice"],
                package_id=backend["package_id"],
            )
            self.records[job_id] = record
            persist_status(folder, record)
            self.queue.put_nowait(job_id)
            return self.public(record)

    def public(self, row):
        result = dict(row)
        if row["status"] == "success":
            result["audio_url"] = "/v1/jobs/" + row["id"] + "/audio"
        return result

    def get(self, job_id):
        if not ID.fullmatch(job_id) or job_id not in self.records:
            raise KeyError(job_id)
        with self.lock:
            return self.public(self.records[job_id])

    def run_process(self, folder):
        command = [
            sys.executable,
            "-B",
            "-m",
            "feiyu_voice_service.worker",
            "--job",
            str(folder / "request.json"),
        ]
        with (folder / "worker.log").open("w", encoding="utf-8") as log:
            with self.lock:
                self.process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                )
            try:
                self.process.wait(timeout=1800)
                if self.process.returncode:
                    raise RuntimeError("Inference failed; inspect the local worker.log")
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=15)
                raise RuntimeError("Inference exceeded the 30 minute job limit")
            finally:
                with self.lock:
                    self.process = None
        result = load(folder / "worker_result.json")
        if (
            result["status"] != "success"
            or sha(folder / "audio.wav") != result["sha256"]
        ):
            raise RuntimeError("Inference output integrity check failed")
        return result

    def work(self):
        while not self.stopping.is_set():
            try:
                job_id = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            folder = self.jobs / job_id
            with self.lock:
                self.active = job_id
                row = self.records[job_id]
                row.update(status="running", started_at=timestamp())
                persist_status(folder, row)
            try:
                result = self.runner(folder)
                with self.lock:
                    row.update(
                        status="success", audio=result["audio"], sha256=result["sha256"]
                    )
            except Exception as exc:
                with self.lock:
                    row.update(status="failed", error=str(exc))
            finally:
                with self.lock:
                    row["finished_at"] = timestamp()
                    persist_status(folder, row)
                    self.active = None
                self.queue.task_done()

    def idle(self):
        return self.active is None and self.queue.empty()
