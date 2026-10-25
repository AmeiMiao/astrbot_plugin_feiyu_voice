import argparse
import json
from pathlib import Path
import sys

from .pipeline import synthesize
from .score import midi_inventory, read_score


def main():
    parser = argparse.ArgumentParser(description="肥鱼按谱唱歌：MIDI/ABC + 中文歌词 → SVS → 最终 A60 Cover 包")
    parser.add_argument("score", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--lyrics")
    source.add_argument("--lyrics-file", type=Path)
    parser.add_argument("--pinyin-file", type=Path, help="可选逐字拼音覆盖（带声调数字），不含延音位")
    parser.add_argument("--track", type=int, help="MIDI 轨道 / ABC 声部，从 0 开始")
    parser.add_argument("--channel", type=int, help="MIDI 通道，从 0 开始")
    parser.add_argument("--midi-encoding", default="utf-8", help="MIDI 文本编码，旧中文文件可用 gbk")
    parser.add_argument("--bpm", type=float, help="可选固定速度覆盖；默认保留谱内速度")
    parser.add_argument("--transpose", type=int, default=0, help="显式移调半音数，默认 0")
    parser.add_argument("--segment-seconds", type=float, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--regions", type=Path, help="本曲 A60 输出对应的人工 D76 换气区间")
    parser.add_argument("--prepare-only", action="store_true", help="仅生成并校验对齐，不启动 GPU")
    parser.add_argument("--inspect", action="store_true", help="查看 MIDI 轨道或 ABC 旋律，不生成")
    args = vars(parser.parse_args())
    try:
        if args.pop("inspect"):
            path = args["score"]
            if path.suffix.lower() in (".mid", ".midi"):
                _, rows = midi_inventory(path, args["midi_encoding"])
                print(json.dumps(rows, ensure_ascii=False, indent=2))
            else:
                notes, duration, info = read_score(path, track=args["track"], channel=args["channel"], bpm=args["bpm"])
                print(json.dumps(dict(info, notes=len(notes), duration_seconds=duration), ensure_ascii=False, indent=2))
        else:
            synthesize(**args)
        return 0
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print(f"唱歌任务未完成：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
