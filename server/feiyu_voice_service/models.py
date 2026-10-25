from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    voice: str = Field(default="feiyu", min_length=1, max_length=80)


class SpeechRequest(Request):
    text: str = Field(min_length=1, max_length=2000)
    style: str = Field(default="cute", min_length=1, max_length=40)

    @model_validator(mode="after")
    def nonempty(self):
        if not self.text.strip():
            raise ValueError("text must contain readable content")
        return self


class ConversionRequest(Request):
    audio_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    regions_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")


class SingingRequest(Request):
    score_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    lyrics: str | None = Field(default=None, max_length=10000)
    lyrics_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    pinyin_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    regions_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    track: int | None = Field(default=None, ge=0)
    channel: int | None = Field(default=None, ge=0, le=15)
    bpm: float | None = Field(default=None, ge=10, le=600, allow_inf_nan=False)
    transpose: int = Field(default=0, ge=-24, le=24)
    midi_encoding: Literal["utf-8", "gbk", "latin1", "shift_jis"] = "utf-8"
    segment_seconds: float = Field(default=20, ge=5, le=25, allow_inf_nan=False)

    @model_validator(mode="after")
    def exclusive_lyrics(self):
        if self.lyrics is not None and self.lyrics_id:
            raise ValueError("Use lyrics or lyrics_id, not both")
        return self
