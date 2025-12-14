# 🎙️ whisper-audio-to-text

Простой скрипт на Python, который **переводит аудио/видео в текст** с помощью модели [Whisper](https://github.com/openai/whisper).

Поддерживает **русский язык**, работает локально, без интернета (после первой загрузки модели).

> **Автор**: Москалев Виталий, РФ  
> **Контакт**: moskalevvitalii@mail.ru  
> **Сообщество**: [vk.com/hihichpock](https://vk.com/hihichpock)  
> **Год**: 2025  
> **Лицензия**: MIT — см. [LICENSE](./LICENSE)

## 📦 Требования

- **Python 3.8+**  
  → Установите с [python.org](https://www.python.org/downloads/)
- **`ffmpeg`**  
  → Скачайте [ffmpeg-release-essentials.zip](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip)  
  → Распакуйте → скопируйте `ffmpeg.exe` в `C:\` (или добавьте в `PATH`)
- **Зависимости**:
  ```bash
  pip install openai-whisper