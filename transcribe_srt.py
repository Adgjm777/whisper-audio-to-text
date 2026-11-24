#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
РўСЂР°РЅСЃРєСЂРёР±Р°С†РёСЏ РІРёРґРµРѕ/Р°СѓРґРёРѕ РІ СЃСѓР±С‚РёС‚СЂС‹ СЃ С‚Р°Р№РјРєРѕРґР°РјРё.

РћСЃРЅРѕРІР°РЅРѕ РЅР° OpenAI Whisper. РџРѕ СѓРјРѕР»С‡Р°РЅРёСЋ РёСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ openai-whisper,
РµСЃР»Рё СѓСЃС‚Р°РЅРѕРІР»РµРЅ faster-whisper вЂ” РјРѕР¶РЅРѕ РІС‹Р±СЂР°С‚СЊ РµРіРѕ С‡РµСЂРµР· --backend faster.

Р’С‹С…РѕРґ: SRT, VTT Рё JSON СЃ СЂРµРїР»РёРєР°РјРё. РџРѕ Р¶РµР»Р°РЅРёСЋ вЂ” РІРёРґРµРѕ СЃ РІР¶Р°С‚С‹РјРё СЃСѓР±С‚РёС‚СЂР°РјРё.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence, Tuple


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Cue:
    start: float
    end: float
    text: str


def find_ffmpeg(explicit: Optional[str] = None) -> str:
    candidates = []
    if explicit:
        candidates.append(explicit)
    env_value = os.environ.get("FFMPEG")
    if env_value:
        candidates.append(env_value)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(script_dir, "ffmpeg.exe"))
    candidates.append(os.path.join(script_dir, "ffmpeg"))
    candidates.append(shutil.which("ffmpeg") or "")
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    raise RuntimeError(
        "ffmpeg РЅРµ РЅР°Р№РґРµРЅ. РџРѕР»РѕР¶РёС‚Рµ ffmpeg.exe СЂСЏРґРѕРј СЃРѕ СЃРєСЂРёРїС‚РѕРј, "
        "СѓРєР°Р¶РёС‚Рµ --ffmpeg РїСѓС‚СЊ/Рє/ffmpeg РёР»Рё РґРѕР±Р°РІСЊС‚Рµ ffmpeg РІ PATH."
    )


def run_command(command: Sequence[str], cwd: Optional[str] = None) -> None:
    process = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.returncode != 0:
        tail = (process.stderr or "").strip().splitlines()[-12:]
        raise RuntimeError(
            "РљРѕРјР°РЅРґР° Р·Р°РІРµСЂС€РёР»Р°СЃСЊ СЃ РѕС€РёР±РєРѕР№:\n"
            + " ".join(command)
            + "\n"
            + "\n".join(tail)
        )


def extract_audio(ffmpeg: str, source: str, destination: str) -> None:
    run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            source,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            destination,
        ]
    )


def _get_value(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _segment_words(segment) -> List[Word]:
    words = _get_value(segment, "words")
    if words:
        result = []
        for word in words:
            text = (_get_value(word, "word", "") or "").strip()
            start = _get_value(word, "start")
            end = _get_value(word, "end")
            if text and start is not None and end is not None:
                result.append(Word(float(start), float(end), text))
        if result:
            return result
    text = (_get_value(segment, "text", "") or "").strip()
    start = _get_value(segment, "start")
    end = _get_value(segment, "end")
    if text and start is not None and end is not None:
        return [Word(float(start), float(end), text)]
    return []


def transcribe_faster(
    audio_path: str,
    model_size: str,
    language: Optional[str],
    device: str,
    compute_type: str,
) -> Tuple[List[Word], str]:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper РЅРµ СѓСЃС‚Р°РЅРѕРІР»РµРЅ. РЈСЃС‚Р°РЅРѕРІРёС‚Рµ: pip install faster-whisper"
        ) from exc

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        audio_path,
        language=language or None,
        vad_filter=True,
        word_timestamps=True,
        beam_size=5,
    )
    words: List[Word] = []
    for segment in segments:
        words.extend(_segment_words(segment))
    return words, getattr(info, "language", "unknown") or "unknown"


def transcribe_openai(
    audio_path: str,
    model_size: str,
    language: Optional[str],
    device: str,
) -> Tuple[List[Word], str]:
    try:
        import whisper  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "openai-whisper РЅРµ СѓСЃС‚Р°РЅРѕРІР»РµРЅ. РЈСЃС‚Р°РЅРѕРІРёС‚Рµ: pip install openai-whisper"
        ) from exc

    model = whisper.load_model(model_size, device=device)
    result = model.transcribe(
        audio_path,
        language=language or None,
        fp16=(device != "cpu"),
        word_timestamps=True,
        verbose=False,
    )
    words: List[Word] = []
    for segment in result.get("segments", []):
        words.extend(_segment_words(segment))
    return words, result.get("language", "unknown") or "unknown"


def _wrap_lines(text: str, max_line: int) -> List[str]:
    return textwrap.wrap(
        text,
        width=max_line,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]


def _would_exceed_lines(words: Sequence[Word], max_line: int, max_lines: int) -> bool:
    text = " ".join(word.text for word in words).strip()
    return len(_wrap_lines(text, max_line)) > max_lines


def _ends_sentence(text: str) -> bool:
    return bool(re.search(r"[.!?вЂ¦]\s*$", text))


def build_cues(
    words: Sequence[Word],
    max_line: int = 42,
    max_lines: int = 2,
    max_duration: float = 6.0,
    max_gap: float = 0.9,
    min_duration: float = 1.2,
) -> List[Cue]:
    cleaned = [w for w in words if w.text.strip() and w.end > w.start]
    if not cleaned:
        return []

    cues: List[Cue] = []
    current: List[Word] = []

    def flush() -> None:
        if not current:
            return
        text = " ".join(word.text for word in current).strip()
        text = re.sub(r"\s+", " ", text)
        if text:
            cues.append(Cue(current[0].start, current[-1].end, text))
        current.clear()

    for word in cleaned:
        if current:
            current_text = " ".join(item.text for item in current).strip()
            gap = word.start - current[-1].end
            too_long = (word.end - current[0].start) > max_duration
            too_wide = _would_exceed_lines(list(current) + [word], max_line, max_lines)
            sentence_break = _ends_sentence(current_text) and len(current_text) >= max_line
            if gap > max_gap or too_long or too_wide or sentence_break:
                flush()
        current.append(word)
    flush()

    for index, cue in enumerate(cues):
        next_start = cues[index + 1].start if index + 1 < len(cues) else None
        minimum_end = cue.start + min_duration
        if cue.end < minimum_end:
            cue.end = minimum_end
        if next_start is not None and cue.end > next_start - 0.05:
            cue.end = max(cue.start + 0.5, next_start - 0.05)
        if cue.end <= cue.start:
            cue.end = cue.start + 0.5

    return cues


def format_time(seconds: float, separator: str = ",") -> str:
    if seconds < 0:
        seconds = 0.0
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def to_srt(cues: Sequence[Cue], max_line: int = 42) -> str:
    blocks: List[str] = []
    for index, cue in enumerate(cues, start=1):
        lines = _wrap_lines(cue.text, max_line)
        blocks.append(
            f"{index}\n"
            f"{format_time(cue.start, ',')} --> {format_time(cue.end, ',')}\n"
            + "\n".join(lines)
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def to_vtt(cues: Sequence[Cue], max_line: int = 42) -> str:
    blocks: List[str] = []
    for cue in cues:
        lines = _wrap_lines(cue.text, max_line)
        blocks.append(
            f"{format_time(cue.start, '.')} --> {format_time(cue.end, '.')}\n"
            + "\n".join(lines)
        )
    body = "\n\n".join(blocks)
    return "WEBVTT\n\n" + (body + "\n" if body else "")


def to_json(cues: Sequence[Cue], language: str) -> str:
    payload = {
        "language": language,
        "cues": [asdict(cue) for cue in cues],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def write_text(path: str, content: str, crlf: bool = True) -> None:
    newline = "\r\n" if crlf else "\n"
    normalized = content.replace("\r\n", "\n").replace("\n", newline)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(normalized)


def burn_subtitles(
    ffmpeg: str,
    video_path: str,
    srt_path: str,
    output_path: str,
    font_name: str = "VK Sans Display",
    font_size: int = 22,
    margin_v: int = 40,
) -> None:
    directory = os.path.dirname(os.path.abspath(srt_path))
    srt_name = os.path.basename(srt_path)
    style = (
        f"FontName={font_name},FontSize={font_size},"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "BorderStyle=1,Outline=2,Shadow=0,Alignment=2,"
        f"MarginV={margin_v}"
    )
    run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            os.path.abspath(video_path),
            "-vf",
            f"subtitles={srt_name}:force_style='{style}'",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            os.path.abspath(output_path),
        ],
        cwd=directory,
    )


def run_self_test() -> int:
    words = [
        Word(0.0, 0.4, "РџСЂРёРІРµС‚,"),
        Word(0.4, 0.9, "СЌС‚Рѕ"),
        Word(0.9, 1.4, "РїСЂРѕРІРµСЂРєР°"),
        Word(1.4, 1.8, "СЃСѓР±С‚РёС‚СЂРѕРІ."),
        Word(3.2, 3.6, "Р’С‚РѕСЂР°СЏ"),
        Word(3.6, 4.1, "СЂРµРїР»РёРєР°"),
        Word(4.1, 4.6, "РїРѕСЃР»Рµ"),
        Word(4.6, 5.0, "РїР°СѓР·С‹."),
    ]
    cues = build_cues(words)
    print(to_srt(cues))
    print(to_vtt(cues))
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Р Р°СЃРїРѕР·РЅР°С‚СЊ РІРёРґРµРѕ/Р°СѓРґРёРѕ Рё СЃРѕС…СЂР°РЅРёС‚СЊ СЃСѓР±С‚РёС‚СЂС‹ СЃ С‚Р°Р№РјРєРѕРґР°РјРё.",
    )
    parser.add_argument("input", nargs="?", help="РІРёРґРµРѕ РёР»Рё Р°СѓРґРёРѕ С„Р°Р№Р»")
    parser.add_argument(
        "--model",
        default="small",
        help="СЂР°Р·РјРµСЂ РјРѕРґРµР»Рё Whisper: tiny, base, small, medium, large, turbo (РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ small)",
    )
    parser.add_argument(
        "--lang",
        default=None,
        help="СЏР·С‹Рє: ru, en Рё С‚.Рґ. Р‘РµР· РїР°СЂР°РјРµС‚СЂР° вЂ” Р°РІС‚РѕРѕРїСЂРµРґРµР»РµРЅРёРµ",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "faster", "openai"),
        default="auto",
        help="РґРІРёР¶РѕРє СЂР°СЃРїРѕР·РЅР°РІР°РЅРёСЏ (РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ auto)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu РёР»Рё cuda (РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ cpu)",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="С‚РёРї РІС‹С‡РёСЃР»РµРЅРёР№ РґР»СЏ faster-whisper: int8, float16 Рё С‚.Рґ.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="РїР°РїРєР° РґР»СЏ СЂРµР·СѓР»СЊС‚Р°С‚РѕРІ (РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ СЂСЏРґРѕРј СЃ РёСЃС…РѕРґРЅРёРєРѕРј)",
    )
    parser.add_argument(
        "--formats",
        default="srt,vtt,json",
        help="С„РѕСЂРјР°С‚С‹ С‡РµСЂРµР· Р·Р°РїСЏС‚СѓСЋ: srt,vtt,json (РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ РІСЃРµ)",
    )
    parser.add_argument(
        "--burn",
        action="store_true",
        help="РІР¶РµС‡СЊ СЃСѓР±С‚РёС‚СЂС‹ РІ РІРёРґРµРѕ",
    )
    parser.add_argument(
        "--burn-out",
        default=None,
        help="РёРјСЏ РІС‹С…РѕРґРЅРѕРіРѕ РІРёРґРµРѕ СЃ СЃСѓР±С‚РёС‚СЂР°РјРё",
    )
    parser.add_argument(
        "--max-line",
        type=int,
        default=42,
        help="РјР°РєСЃРёРјСѓРј СЃРёРјРІРѕР»РѕРІ РІ СЃС‚СЂРѕРєРµ СЃСѓР±С‚РёС‚СЂР° (РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ 42)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=6.0,
        help="РјР°РєСЃРёРјР°Р»СЊРЅР°СЏ РґР»РёС‚РµР»СЊРЅРѕСЃС‚СЊ РѕРґРЅРѕР№ СЂРµРїР»РёРєРё, СЃРµРє (РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ 6)",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=0.9,
        help="РїР°СѓР·Р°, РїРѕСЃР»Рµ РєРѕС‚РѕСЂРѕР№ РЅР°С‡РёРЅР°РµС‚СЃСЏ РЅРѕРІР°СЏ СЂРµРїР»РёРєР°, СЃРµРє (РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ 0.9)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.2,
        help="РјРёРЅРёРјР°Р»СЊРЅР°СЏ РґР»РёС‚РµР»СЊРЅРѕСЃС‚СЊ СЂРµРїР»РёРєРё, СЃРµРє (РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ 1.2)",
    )
    parser.add_argument(
        "--ffmpeg",
        default=None,
        help="РїСѓС‚СЊ Рє ffmpeg (РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ РёС‰РµС‚СЃСЏ СЂСЏРґРѕРј СЃРѕ СЃРєСЂРёРїС‚РѕРј РёР»Рё РІ PATH)",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="РЅРµ СѓРґР°Р»СЏС‚СЊ РІСЂРµРјРµРЅРЅС‹Р№ WAV РїРѕСЃР»Рµ РѕР±СЂР°Р±РѕС‚РєРё",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="РїСЂРѕРІРµСЂРёС‚СЊ СЃР±РѕСЂРєСѓ SRT/VTT Р±РµР· Р·Р°РїСѓСЃРєР° Whisper",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    if not args.input:
        print("РЈРєР°Р¶РёС‚Рµ С„Р°Р№Р». РџСЂРёРјРµСЂ: python transcribe_srt.py video.mp4 --lang ru", file=sys.stderr)
        return 2
    if not os.path.isfile(args.input):
        print(f"Р¤Р°Р№Р» РЅРµ РЅР°Р№РґРµРЅ: {args.input}", file=sys.stderr)
        return 2

    ffmpeg = find_ffmpeg(args.ffmpeg)
    out_dir = os.path.abspath(args.out_dir or os.path.dirname(os.path.abspath(args.input)))
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input))[0]

    temp_dir = tempfile.mkdtemp(prefix="whisper_srt_")
    audio_path = os.path.join(temp_dir, "audio.wav")

    try:
        print(f"[1/3] РР·РІР»РµРєР°СЋ Р·РІСѓРє: {args.input}")
        extract_audio(ffmpeg, args.input, audio_path)

        backend = args.backend
        if backend == "auto":
            backend = "faster" if importlib.util.find_spec("faster_whisper") else "openai"

        print(f"[2/3] Р Р°СЃРїРѕР·РЅР°СЋ ({backend}, РјРѕРґРµР»СЊ {args.model}, СЏР·С‹Рє {args.lang or 'Р°РІС‚Рѕ'})...")
        if backend == "faster":
            words, language = transcribe_faster(
                audio_path, args.model, args.lang, args.device, args.compute_type
            )
        elif backend == "openai":
            words, language = transcribe_openai(
                audio_path, args.model, args.lang, args.device
            )
        else:
            raise RuntimeError(f"РќРµРёР·РІРµСЃС‚РЅС‹Р№ backend: {backend}")

        if not words:
            raise RuntimeError("Р РµС‡СЊ РЅРµ СЂР°СЃРїРѕР·РЅР°РЅР°: РІ С„Р°Р№Р»Рµ РЅРµС‚ СЂРµС‡Рё РёР»Рё Р·РІСѓРє СЃР»РёС€РєРѕРј С‚РёС…РёР№.")

        cues = build_cues(
            words,
            max_line=args.max_line,
            max_duration=args.max_duration,
            max_gap=args.max_gap,
            min_duration=args.min_duration,
        )
        if not cues:
            raise RuntimeError("РќРµ СѓРґР°Р»РѕСЃСЊ СЃРѕР±СЂР°С‚СЊ СЂРµРїР»РёРєРё РґР»СЏ СЃСѓР±С‚РёС‚СЂРѕРІ.")

        formats = {item.strip().lower() for item in args.formats.split(",") if item.strip()}
        created = []
        if "srt" in formats:
            srt_path = os.path.join(out_dir, base_name + ".srt")
            write_text(srt_path, to_srt(cues, args.max_line), crlf=True)
            created.append(srt_path)
        else:
            srt_path = None
        if "vtt" in formats:
            vtt_path = os.path.join(out_dir, base_name + ".vtt")
            write_text(vtt_path, to_vtt(cues, args.max_line), crlf=True)
            created.append(vtt_path)
        if "json" in formats:
            json_path = os.path.join(out_dir, base_name + ".json")
            write_text(json_path, to_json(cues, language), crlf=False)
            created.append(json_path)

        if args.burn:
            if not srt_path:
                srt_path = os.path.join(out_dir, base_name + ".srt")
                write_text(srt_path, to_srt(cues, args.max_line), crlf=True)
                created.append(srt_path)
            burned_path = args.burn_out or os.path.join(out_dir, base_name + "_subtitled.mp4")
            print("[3/3] Р’Р¶РёРіР°СЋ СЃСѓР±С‚РёС‚СЂС‹...")
            burn_subtitles(
                ffmpeg,
                args.input,
                srt_path,
                burned_path,
                font_size=max(18, int(args.max_line // 2)),
                margin_v=40,
            )
            created.append(burned_path)

        print(f"РЇР·С‹Рє: {language}")
        print(f"Р РµРїР»РёРє: {len(cues)}")
        print("Р“РѕС‚РѕРІРѕ:")
        for path in created:
            print(f"  {path}")
        return 0
    finally:
        if args.keep_audio:
            print(f"Р’СЂРµРјРµРЅРЅС‹Р№ Р·РІСѓРє РѕСЃС‚Р°РІР»РµРЅ: {audio_path}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("РћСЃС‚Р°РЅРѕРІР»РµРЅРѕ РїРѕР»СЊР·РѕРІР°С‚РµР»РµРј.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:  # noqa: BLE001
        print(f"РћС€РёР±РєР°: {error}", file=sys.stderr)
        sys.exit(1)