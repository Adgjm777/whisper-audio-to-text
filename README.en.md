# whisper-audio-to-text

A small Python script that turns audio or video into text, locally on your own machine.
It uses OpenAI Whisper and needs the internet only once, to download the model.

Defaults are configured for Russian speech with the `small` model. Changing the model and the
language is a one-line edit, see below.

> **Author**: Vitalii Moskalev
> **Contact**: moskalevvitalii@mail.ru
> **Community**: https://vk.com/hihichpock
> **Year**: 2025
> **License**: MIT, see LICENSE

---

## Features

- Transcribes speech from video and audio: `mp4`, `mkv`, `mov`, `mp3`, `wav`, `m4a`, anything ffmpeg can read.
- Writes the result next to the source file: `video.mp4` becomes `video.txt`.
- Works offline after the first model download.
- Nothing is uploaded anywhere: the transcript stays on your computer.

---

## Requirements

- **Python 3.8 or newer** — https://www.python.org/downloads/
- **ffmpeg** — Whisper cannot decode files without it.
- **The Whisper package** — one pip install, PyTorch comes with it.

---

## Install on Windows

1. Install Python and tick **Add Python to PATH** during setup.
2. Download ffmpeg: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
   Unpack it, find `ffmpeg.exe` inside and copy it to `C:\`, or add its folder to `PATH`.
3. Install the package:

```bash
pip install openai-whisper
```

4. Check that ffmpeg is visible:

```bash
ffmpeg -version
```

If the command is not found, `ffmpeg.exe` is not where the system looks for it.

---

## Usage

```bash
python transcribe.py C:\video\interview.mp4
```

What happens:

1. The first run downloads the `small` model (about 470 MB), later runs use the cache.
2. The recognised text is printed to the console.
3. A file `interview.txt` appears next to the source file.

---

## Models

The model is set in `transcribe.py`:

```python
model = whisper.load_model("small")
```

| Model | Size | Speed on CPU | Quality |
|---|---|---|---|
| `tiny` | ~75 MB | fastest | rough, many errors |
| `base` | ~140 MB | fast | acceptable for clean speech |
| `small` | ~470 MB | medium | good, the default |
| `medium` | ~1.5 GB | slow | very good |
| `large` | ~3 GB | very slow | best |

Rule of thumb: `medium` for interviews and lectures, `small` is enough for short clips.
On a weak laptop `large` will take hours.

The language is set next to the transcribe call:

```python
result = model.transcribe(file, language="ru", fp16=False)
```

`language="ru"` means Russian. Use another code for another language: `en`, `uk`, `de`, `fr`
and so on. If you do not know the language, pass `language=None` and Whisper will detect it,
with lower accuracy.

`fp16=False` is the CPU mode. On an NVIDIA GPU you can set it to `True` for speed.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ffmpeg not found` | ffmpeg missing or not in PATH | copy `ffmpeg.exe` to `C:\` or add it to `PATH` |
| `No module named whisper` | package not installed | `pip install openai-whisper` |
| very slow | large model or no GPU | use `small` or `base` |
| `CUDA out of memory` | not enough GPU memory | keep `fp16=False` and pick a smaller model |
| `File not found` | wrong path | wrap paths with spaces in quotes |

---

## Limitations

- One file per run, no batch folder processing.
- No timestamps: the output is plain text.
- Quality drops on noisy recordings, music and overlapping speakers.

---

## License

MIT. Commercial use, modification and redistribution are allowed.
The only condition is to keep the license text and the author notice. See LICENSE.