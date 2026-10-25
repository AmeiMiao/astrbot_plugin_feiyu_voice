"""Recreate the tiny original MIDI fixture, including Chinese lyric/continuation events."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import mido
from feiyu_singing.score import lyric_tokens, read_score


def main():
    folder = Path(__file__).resolve().parent
    notes, duration, _ = read_score(folder / "melisma_demo.abc")
    lyrics = lyric_tokens((folder / "melisma_demo_lyrics.txt").read_text(encoding="utf-8"))
    song = mido.MidiFile(type=1, ticks_per_beat=480, charset="utf-8")
    conductor = mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=750000, time=0)])
    track = mido.MidiTrack([mido.MetaMessage("track_name", name="Original melisma demo", time=0)])
    song.tracks.extend([conductor, track])
    events = []
    for note, text in zip(notes, lyrics):
        a, b = round(note.start * 640), round(note.end * 640)
        events.extend([(a, 1, mido.MetaMessage("lyrics", text=text)),
                       (a, 2, mido.Message("note_on", note=note.pitch, velocity=80)),
                       (b, 0, mido.Message("note_off", note=note.pitch, velocity=0))])
    previous = 0
    for tick, _, message in sorted(events, key=lambda item: (item[0], item[1])):
        track.append(message.copy(time=tick - previous))
        previous = tick
    track.append(mido.MetaMessage("end_of_track", time=round(duration * 640) - previous))
    output = folder / "melisma_demo.mid"
    if output.exists():
        raise FileExistsError("Demo MIDI already exists; it has not been overwritten")
    song.save(output)
    print(output)


if __name__ == "__main__":
    main()
