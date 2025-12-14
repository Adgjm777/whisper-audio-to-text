# whisper-audio-to-text
# Автор: Москалев Виталий, РФ
# Email: moskalevvitalii@mail.ru
# Сообщество: https://vk.com/hihichpock
# Год: 2025
# Лицензия: MIT (см. LICENSE)
# ---------------------------------------------

import whisper
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Использование: python transcribe.py путь/к/файлу.mp4")
        sys.exit(1)

    file = sys.argv[1]
    if not os.path.exists(file):
        print(f"❌ Файл не найден: {file}")
        sys.exit(1)

    print("📥 Загружаю модель 'small' (русский, ~90% точность)...")
    model = whisper.load_model("small")

    print(f"🔊 Распознаю: {file}")
    result = model.transcribe(file, language="ru", fp16=False)

    text = result["text"].strip()
    print("\n✅ Текст:\n")
    print(text)

    # Сохранить в файл
    out_file = os.path.splitext(file)[0] + ".txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n💾 Сохранено: {out_file}")

if __name__ == "__main__":
    main()