# Audio-Intelligence-MCP
🎧 A modular FastMCP server for audio processing, transcription, classification, and analysis. Includes automatic feature extraction, noise reduction, speaker diarization, and more.

Ses işleme için MCP (Model Context Protocol) tabanlı bir sunucu. Transkripsiyon, özellik analizi, sınıflandırma, metadata çıkarımı ve format dönüştürme araçları sağlar.

## İçerik
- Kurulum
- Çalıştırma
- Araçlar (Tools)
- Örnek Kullanımlar
- Çıktılar ve Klasör Yapısı
- Notlar ve İpuçları

## Kurulum
1) Python 3.10+ ve (opsiyonel) CUDA destekli PyTorch kurulu olmalı.
2) Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```
3) pydub için FFmpeg gerekli. Sisteminizde FFmpeg kurulu olduğundan emin olun ve PATH'e ekleyin.
   - Windows: `choco install ffmpeg` veya `winget install Gyan.FFmpeg`
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg`

## Çalıştırma
### MCP Sunucusunu Stdio ile Başlatma
Sunucu, `stdio` transport ile çalışır ve bir MCP istemcisi tarafından başlatılır. Doğrudan test etmek için:
```bash
python main.py
```
Bu modda bir MCP istemcisi olmadan anlamlı bir çıktı vermez; tipik kullanım bir istemci aracılığıyladır.

### Örnek İstemci (LangGraph + Gemini)
`try.py`, `main.py` MCP sunucusunu `stdio` ile başlatır, araçları keşfeder ve ReAct ajanına tanımlar:
```bash
python try.py
```
Varsayılan örnek, `audio/` altındaki dosyalarla bir transkripsiyon isteği yapar.

## Araçlar (Tools)
Sunucu adı: `Audio-Intelligence-MCP`

Tüm araçlar dosya yolu listesi alır. Geçersiz yol veya desteklenmeyen formatta hata döner. Desteklenen formatlar: `.wav`, `.mp3`, `.ogg`, `.flac`.

### transcript
- Amaç: Ses dosyalarını metne çevirir (Whisper base).
- Girdi:
  - `file_paths: List[str]`
  - `language: str = "en"`
  - `output_dir: str = "output"`
  - `overwrite: bool = False`
- Çıktı:
  - `{ "transcripts": { "<path>": { "transcription": str, "transcript_file": str } | { "error": str } } }`
- Not: 16kHz örnekleme ile işler ve çıktıyı `output/transcripts/` altına yazar.

### feature_analysis
- Amaç: Pitch, tempo, süre gibi temel özellikleri çıkarır; dalga formu PNG üretir.
- Girdi:
  - `file_paths: List[str]`
  - `output_dir: str = "output"`
  - `overwrite: bool = False`
- Çıktı:
  - `{ "analyses": { "<path>": { "features": { "mean_pitch": float, "tempo": float, "duration_s": float }, "waveform_plot": str } | { "error": str } } }`
- Not: Dalga formu görselleri `output/waveforms/` altına kaydedilir.

### audio_classification
- Amaç: Ses içeriğini sınıflandırır (AST, AudioSet üstünde eğitilmiş model).
- Girdi:
  - `file_paths: List[str]`
  - `output_dir: str = "output"`
  - `output_csv: Optional[str] = None`
  - `overwrite: bool = False`
- Çıktı:
  - `{ "classifications": { "<path>": { "label": str, "confidence": float } | { "error": str }, "csv_path"?: str, "csv_error"?: str } }`
- Not: `output/classifications/` içine opsiyonel CSV yazar.

### metadata_extraction
- Amaç: Süre, örnek hızı, kanal sayısı, etiketler (tags) gibi metadata çıkarır ve JSON kaydeder.
- Girdi:
  - `file_paths: List[str]`
  - `output_dir: str = "output"`
  - `overwrite: bool = False`
- Çıktı:
  - `{ "metadata": { "<path>": { "metadata": { "duration_ms": int, "bitrate": int, "channels": int, "tags": object }, "metadata_file": str } | { "error": str } } }`
- Not: JSON dosyaları `output/metadata/` altına yazılır.

### audio_conversion
- Amaç: Dosyaları hedef ses formatına dönüştürür.
- Girdi:
  - `file_paths: List[str]`
  - `target_format: str = "wav"`
  - `output_dir: str = "output"`
  - `overwrite: bool = False`
- Çıktı:
  - `{ "converted_files": { "<path>": { "converted_file": str } | { "error": str } } }`
- Not: Çıktılar `output/converted/` altına yazılır.

## Örnek Kullanımlar
Aşağıdaki örnekler MCP istemcisi tarafından tool çağrısına dönüştürülür. Pseudo girdi örnekleri:

```json
{
  "tool": "transcript",
  "args": { "file_paths": ["audio/speech-94649.wav"], "language": "en" }
}
```

```json
{
  "tool": "feature_analysis",
  "args": { "file_paths": ["audio/speech-94649.wav"] }
}
```

```json
{
  "tool": "audio_classification",
  "args": { "file_paths": ["audio/speech-94649.wav"], "output_csv": "labels.csv" }
}
```

```json
{
  "tool": "metadata_extraction",
  "args": { "file_paths": ["audio/speech-94649.wav"] }
}
```

```json
{
  "tool": "audio_conversion",
  "args": { "file_paths": ["audio/speech-94649.wav"], "target_format": "mp3" }
}
```

## Çıktılar ve Klasör Yapısı
- `output/`
  - `transcripts/`: `*_transcript_<timestamp>.txt`
  - `waveforms/`: `*_waveform_<timestamp>.png`
  - `classifications/`: `labels.csv` (opsiyonel)
  - `metadata/`: `*_metadata_<timestamp>.json`
  - `converted/`: `*_converted_<timestamp>.<ext>`

## Notlar ve İpuçları
- GPU varsa otomatik `cuda` kullanılacaktır; aksi halde `cpu`.
- Büyük modeller ilk çağrıda indirileceğinden ilk çalışma süresi uzun olabilir.
- `config.json` dosyası mevcutsa `output_dir`, `sample_rate`, `overwrite_files` gibi varsayılanları günceller.
- Desteklenmeyen formatlar veya mevcut olmayan yollar için araçlar hata mesajı döndürür.

