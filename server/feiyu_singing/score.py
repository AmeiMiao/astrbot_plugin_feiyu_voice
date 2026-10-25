"""Read a monophonic MIDI/ABC score and compile explicit note/lyric alignment."""
from bisect import bisect_right
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
import unicodedata


@dataclass
class Note:
    start: float
    end: float
    pitch: int
    lyric: str = ""


def tempo_clock(changes, units_per_quarter):
    """Create a piecewise tempo clock; tempo is microseconds per quarter."""
    changes = [(0, 500000), *sorted(changes)]
    times, seconds, tempos = [], [], []
    elapsed, previous, last_tempo = 0., 0., 500000
    for position, tempo in changes:
        if not math.isfinite(tempo) or tempo <= 0 or position < 0:
            raise ValueError("谱子的速度必须是正数。")
        elapsed += (position - previous) / units_per_quarter * last_tempo / 1e6
        if times and times[-1] == position:
            tempos[-1] = tempo
        else:
            times.append(position)
            seconds.append(elapsed)
            tempos.append(tempo)
        previous, last_tempo = position, tempo

    def clock(position):
        i = bisect_right(times, position) - 1
        return seconds[i] + (position - times[i]) / units_per_quarter * tempos[i] / 1e6
    return clock


def midi_inventory(path, encoding="utf-8"):
    import mido
    try:
        mid = mido.MidiFile(path, charset=encoding)
    except UnicodeDecodeError as exc:
        raise ValueError("MIDI 文本编码无法解读。旧中文 MIDI 可尝试 --midi-encoding gbk。") from exc
    if mid.type == 2 or mid.ticks_per_beat <= 0:
        raise ValueError("仅支持 PPQ 计时的 MIDI type 0/1，不支持 type 2 或 SMPTE。")
    rows = []
    for index, track in enumerate(mid.tracks):
        channels = sorted({m.channel for m in track if m.type == "note_on" and m.velocity and m.channel != 9})
        for channel in channels:
            rows.append(dict(track=index, channel=channel, name=track.name,
                             notes=sum(m.type == "note_on" and m.velocity > 0 and m.channel == channel for m in track)))
    return mid, rows


def read_midi(path, track=None, channel=None, bpm=None, midi_encoding="utf-8"):
    mid, rows = midi_inventory(path, midi_encoding)
    options = [r for r in rows if (track is None or r["track"] == track) and
               (channel is None or r["channel"] == channel)]
    if len(options) != 1:
        raise ValueError(f"需要选定唯一主旋律轨道和通道；请用 --inspect 查看，再指定 --track / --channel。可选：{rows}")
    selected = options[0]
    tempos = {}
    for tr in mid.tracks:
        tick = 0
        for msg in tr:
            tick += msg.time
            if msg.type == "set_tempo":
                if tick in tempos and tempos[tick] != msg.tempo:
                    raise ValueError("MIDI 同一时刻存在冲突的速度事件。")
                tempos[tick] = msg.tempo
    if bpm is not None:
        tempos = {0: 60e6 / bpm}
    clock = tempo_clock(list(tempos.items()), mid.ticks_per_beat)
    tick, active, notes, lyrics = 0, {}, [], {}
    for msg in mid.tracks[selected["track"]]:
        tick += msg.time
        if msg.type == "lyrics":
            if tick in lyrics:
                raise ValueError("同一音符位置有多份 MIDI 歌词，请提供独立歌词文件。")
            lyrics[tick] = msg.text.strip()
        if msg.type not in ("note_on", "note_off") or msg.channel != selected["channel"]:
            continue
        if msg.type == "note_on" and msg.velocity:
            if msg.note in active:
                raise ValueError("同音高音符重叠，无法确定主旋律时值。")
            active[msg.note] = tick
        else:
            if msg.note not in active:
                raise ValueError("MIDI 存在未配对的 note_off。")
            start = active.pop(msg.note)
            notes.append(Note(clock(start), clock(tick), msg.note, lyrics.get(start, "")))
    if active:
        raise ValueError("MIDI 存在未结束的音符。")
    # Lyric meta events may follow note_on at the same tick.
    for n in notes:
        for position, text in lyrics.items():
            if abs(clock(position) - n.start) < 1e-8:
                n.lyric = text
                break
    return sorted(notes, key=lambda n: n.start), clock(tick), dict(format="midi", selected=selected,
                                                                   tracks=rows, tempo_override_bpm=bpm, text_encoding=midi_encoding)


def read_abc(path, track=None, channel=None, bpm=None):
    from music21 import chord, converter, note, stream, tempo
    if channel is not None:
        raise ValueError("ABC 使用 --track 选择声部，不使用 MIDI channel。")
    text = Path(path).read_text(encoding="utf-8-sig")
    # music21 can silently drop repeat bars in short ABC streams. Do not guess
    # their playback order or pretend that the parsed stream expanded them.
    body = "\n".join(line.split("%", 1)[0] for line in text.splitlines()
                     if not re.match(r"^[A-Za-z]:", line))
    if re.search(r"\|:|:\||::|\[[0-9]|\|[0-9]", body):
        raise ValueError("第一版 ABC 请先展开反复记号和跳房子，写出实际演唱顺序。")
    parsed = converter.parseData(text, format="abc")
    if isinstance(parsed, stream.Opus):
        raise ValueError("一个 ABC 文件只能包含一首曲子，请拆分 X: 曲目。")
    parts = list(parsed.parts) or [parsed]
    if track is None:
        if len(parts) != 1:
            raise ValueError(f"ABC 有 {len(parts)} 个声部，请指定 --track（从 0 开始）。")
        track = 0
    if not 0 <= track < len(parts):
        raise ValueError("ABC 声部编号超出范围。")
    part = parts[track]
    flat = part.flatten()
    changes = []
    for mark in flat.getElementsByClass(tempo.MetronomeMark):
        value = mark.getQuarterBPM()
        if value:
            changes.append((float(mark.offset), 60e6 / value))
    if bpm is not None:
        changes = [(0, 60e6 / bpm)]
    clock = tempo_clock(changes, 1)
    notes, tied = [], False
    for element in flat.notesAndRests:
        if isinstance(element, chord.Chord):
            raise ValueError("ABC 主旋律含和弦，请只保留单音主旋律。")
        if not isinstance(element, note.Note):
            continue
        if element.duration.isGrace or element.quarterLength <= 0:
            raise ValueError("第一版暂不支持零时长装饰音，请改成明确时值的音符。")
        pitch = float(element.pitch.ps)
        if not pitch.is_integer():
            raise ValueError("第一版不支持微分音。")
        start = clock(float(element.offset))
        end = clock(float(element.offset + element.quarterLength))
        tie = element.tie.type if element.tie else None
        if tied:
            if tie not in ("continue", "stop") or notes[-1].pitch != pitch or abs(notes[-1].end - start) > 1e-7:
                raise ValueError("ABC 连音线不完整或跨越休止。")
            notes[-1].end = end
        else:
            if tie in ("continue", "stop"):
                raise ValueError("ABC 连音线缺少起始音符。")
            notes.append(Note(start, end, int(pitch), element.lyric or ""))
        tied = tie in ("start", "continue")
    if tied:
        raise ValueError("ABC 连音线未结束。")
    # w: association varies by parser version. Require external lyrics if none were attached.
    return sorted(notes, key=lambda n: n.start), clock(float(part.highestTime)), dict(
        format="abc", selected=dict(track=track), parts=len(parts), tempo_override_bpm=bpm,
        embedded_lyrics_available=bool(notes) and all(n.lyric for n in notes))


def read_score(path, **kwargs):
    bpm = kwargs.get("bpm")
    if bpm is not None and (not math.isfinite(bpm) or not 10 <= bpm <= 600):
        raise ValueError("BPM 必须在 10–600 之间。")
    suffix = Path(path).suffix.lower()
    if suffix in (".mid", ".midi"):
        result = read_midi(path, **kwargs)
    elif suffix == ".abc":
        kwargs.pop("midi_encoding", None)
        result = read_abc(path, **kwargs)
    else:
        raise ValueError("谱子格式须为 .mid、.midi 或 .abc。")
    notes, duration, info = result
    if not notes or not math.isfinite(duration) or duration > 600:
        raise ValueError("谱子须包含音符，第一版最长 10 分钟。")
    end = 0.
    for n in notes:
        if not all(math.isfinite(v) for v in (n.start, n.end)) or not 0 <= n.start < n.end or not 1 <= n.pitch <= 127:
            raise ValueError("谱子包含非法音高或时值。")
        if n.start < end - 1e-7:
            raise ValueError("主旋律含同时发声的音符，请选择单音旋律轨道或声部。")
        end = n.end
    return notes, max(duration, end), info


def lyric_tokens(text):
    """One Han character per syllable; underscore/tilde means continuation."""
    result = []
    for char in text:
        if '\u3400' <= char <= '\u9fff' or char in "_~":
            result.append(char)
        elif char.isspace() or unicodedata.category(char).startswith("P"):
            continue
        else:
            raise ValueError(f"自动读音当前支持中文；不支持字符 {char!r}。请将数字写成汉字。")
    return result


def align_notes(notes, lyrics=None, pinyin=None):
    from pypinyin import Style, lazy_pinyin
    if lyrics is None:
        if not all(n.lyric for n in notes):
            raise ValueError("谱子没有完整歌词。请提供 --lyrics 或 --lyrics-file；ABC 的 w: 未被解析时也需单独提供。")
        tokens = []
        for n in notes:
            one = lyric_tokens(n.lyric)
            if len(one) != 1:
                raise ValueError("每个 MIDI/ABC 歌词事件须为一个汉字或延音标记 _。")
            tokens += one
    else:
        tokens = lyric_tokens(lyrics)
    if len(tokens) != len(notes):
        raise ValueError(f"歌词与音符数量不同：{len(tokens)} 个字/延音标记，{len(notes)} 个音符。每个音符对应一字；一字多音请用 _ 占位。")
    syllables = [t for t in tokens if t not in "_~"]
    if pinyin is not None:
        phones = pinyin.split()
        if len(phones) != len(syllables) or any(not re.fullmatch(r"[a-zv]+[1-5]", s) for s in phones):
            raise ValueError("拼音文件须为带 1–5 声调数字的拼音，数量等于非延音汉字数。")
    else:
        phones = lazy_pinyin("".join(syllables), style=Style.TONE3, neutral_tone_with_five=True, errors="exception")
    phone_index, previous, result = 0, None, []
    for n, token in zip(notes, tokens):
        if token in "_~":
            if previous is None or n.start - result[-1]["end"] > 1e-7:
                raise ValueError("延音 _ 必须紧接前一个唱字，不能在开头或跨越休止。")
            char, phone = previous
            kind = 3
        else:
            char, phone, kind = token, "zh_" + phones[phone_index], 2
            phone_index += 1
            previous = char, phone
        result.append(dict(**asdict(n), text=char, phoneme=phone, note_type=kind))
    return result


def compile_score(notes, duration, lyrics=None, pinyin=None, transpose=0, segment_seconds=20):
    """Quantize boundaries to the model's 20 ms grid, then split at syllable boundaries."""
    if not 5 <= segment_seconds <= 25:
        raise ValueError("分段长度须为 5–25 秒。")
    aligned = align_notes(notes, lyrics, pinyin)
    events, cursor, max_error = [], 0, 0.
    for n in aligned:
        a, b = round(n["start"] * 50), round(n["end"] * 50)
        max_error = max(max_error, abs(a / 50 - n["start"]), abs(b / 50 - n["end"]))
        if b - a < 2:
            raise ValueError("音符在 20ms 网格上短于 40ms，请放慢速度或简化装饰音。")
        if a > cursor:
            if n["note_type"] == 3:
                raise ValueError("延音不能跨越休止。")
            events.append(dict(start_frame=cursor, end_frame=a, pitch=0, text="<SP>", phoneme="<SP>", note_type=1))
        pitch = n["pitch"] + transpose
        if not 1 <= pitch <= 127:
            raise ValueError("移调后音高超出 MIDI 1–127。")
        events.append(dict(start_frame=a, end_frame=b, pitch=pitch, text=n["text"],
                           phoneme=n["phoneme"], note_type=n["note_type"]))
        cursor = b
    final = max(cursor, round(duration * 50))
    if final > cursor:
        events.append(dict(start_frame=cursor, end_frame=final, pitch=0, text="<SP>", phoneme="<SP>", note_type=1))
    # Split long rests so silence never forces an oversized model segment.
    expanded = []
    limit = int(segment_seconds * 50)
    for e in events:
        while e["note_type"] == 1 and e["end_frame"] - e["start_frame"] > limit:
            expanded.append(dict(e, end_frame=e["start_frame"] + limit))
            e = dict(e, start_frame=e["start_frame"] + limit)
        expanded.append(e)
    groups, start = [], 0
    while start < len(expanded):
        end = start + 1
        while end < len(expanded) and expanded[end]["end_frame"] - expanded[start]["start_frame"] <= limit:
            end += 1
        while end < len(expanded) and expanded[end]["note_type"] == 3:
            end -= 1
        if end <= start or expanded[end - 1]["end_frame"] - expanded[start]["start_frame"] > limit:
            raise ValueError("单个唱字/延音组超过分段上限，请缩短长音或增大分段长度。")
        groups.append(expanded[start:end])
        start = end
    metadata = []
    for i, group in enumerate(groups):
        # The pinned upstream floors sum(duration)*50. A sub-nanosecond closure
        # avoids losing a frame through binary floating point; no audible time change.
        durations = [(e["end_frame"] - e["start_frame"]) / 50 for e in group]
        durations[-1] += 1e-9
        metadata.append(dict(index=f"feiyu_{i:04d}", language="Mandarin",
                             time=[group[0]["start_frame"] * 20, group[-1]["end_frame"] * 20],
                             text=" ".join(e["text"] for e in group), phoneme=" ".join(e["phoneme"] for e in group),
                             note_pitch=" ".join(str(e["pitch"]) for e in group),
                             note_type=" ".join(str(e["note_type"]) for e in group),
                             duration=" ".join(f"{d:.9f}" for d in durations)))
    return dict(metadata=metadata, events=events, notes=aligned, frames=final, duration_seconds=final / 50,
                transpose_semitones=transpose, max_boundary_quantization_error_seconds=max_error,
                alignment="one Han character per note; explicit _ continuation; 20ms grid",
                pinyin_source="explicit override" if pinyin else "pypinyin phrase prediction; pronunciation review recommended")
