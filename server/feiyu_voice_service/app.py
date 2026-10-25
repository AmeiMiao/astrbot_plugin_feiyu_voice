import hashlib
import os
import queue
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from feiyu_singing.pipeline import ROOT, resolve, save

from . import __version__
from .jobs import JobManager
from .models import ConversionRequest, SingingRequest, SpeechRequest
from .registry import Registry


def create_app(config_path=None, *, api_key=None, runner=None):
    registry = Registry(config_path or ROOT / "configs/voice_service.json")
    data = resolve(registry.config["data_dir"])
    manager = JobManager(data, registry.config.get("max_queue", 16), runner=runner)
    key_file = data / "api_key.txt"
    token = api_key or os.environ.get("FEIYU_API_KEY")
    if not token:
        if not key_file.exists():
            with key_file.open("x", encoding="utf-8") as stream:
                stream.write(secrets.token_urlsafe(32))
        token = key_file.read_text(encoding="utf-8").strip()
    if not token or len(token) < 16:
        raise ValueError("The API key must contain at least 16 characters")

    @asynccontextmanager
    async def lifespan(app):
        # An OS-held advisory lock is released even after a process crash.
        # A second process may not manage the same queue directory.
        with (data / "service.lock").open("a+b") as lock:
            lock.seek(0)
            if lock.read(1) == b"":
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            manager.start()
            try:
                yield
            finally:
                manager.stop()

    app = FastAPI(
        title="Feiyu Voice Service",
        version=__version__,
        lifespan=lifespan,
        description="Local asynchronous speech, voice conversion and score singing gateway",
    )
    app.state.manager = manager
    app.state.registry = registry
    app.state.key_file = key_file
    bearer = HTTPBearer(auto_error=False)

    def authenticate(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ):
        if not credentials or not secrets.compare_digest(
            credentials.credentials, token
        ):
            raise HTTPException(
                401,
                "A valid Bearer API key is required",
                headers={"WWW-Authenticate": "Bearer"},
            )

    router = APIRouter(prefix="/v1", dependencies=[Depends(authenticate)])

    @app.get("/health")
    def health():
        return dict(status="ok", version=__version__)

    @router.get("/capabilities")
    def capabilities():
        with manager.lock:
            return dict(
                app.state.registry.public(),
                queue=dict(waiting=manager.queue.qsize(), running=bool(manager.active)),
            )

    @router.post("/assets", status_code=201)
    async def upload(file: UploadFile):
        suffix = Path(file.filename or "").suffix.lower()
        allowed = {".wav", ".flac", ".mid", ".midi", ".abc", ".txt", ".json"}
        if suffix not in allowed:
            raise HTTPException(
                415, "Supported uploads: WAV, FLAC, MIDI, ABC, TXT and JSON"
            )
        identifier = uuid.uuid4().hex
        path = manager.assets / (identifier + suffix)
        limit = int(app.state.registry.config.get("max_upload_mb", 128)) * 1024**2
        digest, size = hashlib.sha256(), 0
        try:
            with path.open("xb") as stream:
                while block := await file.read(1024**2):
                    size += len(block)
                    if size > limit:
                        raise HTTPException(
                            413, "Uploaded file exceeds the configured size limit"
                        )
                    stream.write(block)
                    digest.update(block)
            if not size:
                raise HTTPException(422, "Empty files are not accepted")
            meta = dict(
                id=identifier,
                stored_name=path.name,
                bytes=size,
                sha256=digest.hexdigest(),
                suffix=suffix,
            )
            save(manager.assets / (identifier + ".json"), meta)
            return {k: v for k, v in meta.items() if k != "stored_name"}
        except Exception:
            path.unlink(missing_ok=True)
            raise
        finally:
            await file.close()

    def submit(kind, request):
        try:
            with manager.lock:
                backend = app.state.registry.select(kind, request.voice)
                if kind == "speech" and request.style not in backend["styles"]:
                    raise ValueError(
                        "Requested TTS style is not provided by the installed package"
                    )
                return manager.submit(kind, request.model_dump(), backend)
        except queue.Full:
            raise HTTPException(503, "Inference queue is full; retry later")
        except (OSError, ValueError) as exc:
            # Do not return private configuration paths or tracebacks to API callers.
            raise HTTPException(
                422,
                "Unavailable capability or invalid uploaded asset: "
                + (str(exc) if isinstance(exc, ValueError) else "asset not found"),
            )

    @router.post("/speech", status_code=202)
    def speech(request: SpeechRequest):
        return submit("speech", request)

    @router.post("/voice-conversion", status_code=202)
    def conversion(request: ConversionRequest):
        return submit("conversion", request)

    @router.post("/singing", status_code=202)
    def singing(request: SingingRequest):
        return submit("singing", request)

    @router.get("/jobs/{job_id}")
    def job(job_id: str):
        try:
            return manager.get(job_id)
        except KeyError:
            raise HTTPException(404, "Job not found")

    @router.get("/jobs/{job_id}/audio", response_class=FileResponse)
    def audio(job_id: str):
        row = job(job_id)
        if row["status"] != "success":
            raise HTTPException(409, "Audio is not ready")
        path = (manager.jobs / job_id / "audio.wav").resolve()
        if not path.is_relative_to(manager.jobs) or not path.is_file():
            raise HTTPException(404, "Audio file is unavailable")
        return FileResponse(
            path,
            media_type="audio/wav",
            filename=job_id + ".wav",
            headers={"Cache-Control": "no-store"},
        )

    @router.post("/admin/reload")
    def reload_packages():
        with manager.lock:
            if not manager.idle():
                raise HTTPException(
                    409,
                    "Wait until all queued and running jobs finish before replacing packages",
                )
            try:
                candidate = Registry(app.state.registry.path)
                if (
                    resolve(candidate.config["data_dir"]) != manager.root
                    or candidate.config.get("max_queue", 16) != manager.queue.maxsize
                ):
                    raise ValueError(
                        "Changing queue storage or capacity requires a service restart"
                    )
            except (ValueError, OSError, KeyError):
                raise HTTPException(
                    422,
                    "New package configuration is invalid; current configuration was retained",
                )
            app.state.registry = candidate
            return dict(status="reloaded", capabilities=candidate.public())

    app.include_router(router)
    return app
