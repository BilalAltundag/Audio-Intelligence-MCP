# Audio-Intelligence-MCP
🎧 A modular FastMCP server for audio processing, transcription, classification, and analysis. Includes automatic feature extraction, noise reduction, speaker diarization, and more.

An MCP (Model Context Protocol) based server for audio processing. It provides tools for transcription, feature analysis, classification, metadata extraction, and format conversion.

## Table of Contents
- Installation
- Running
- Tools
- Example Usage
- Outputs and Directory Structure
- Notes and Tips

## Installation
1) Python 3.10+ and (optional) CUDA-enabled PyTorch should be installed.
2) Install dependencies:
```bash
pip install -r requirements.txt
```
3) FFmpeg is required for pydub. Ensure FFmpeg is installed on your system and added to PATH.
   - Windows: `choco install ffmpeg` or `winget install Gyan.FFmpeg`
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt-get install ffmpeg`

## Running
### Start MCP Server via Stdio
The server runs with `stdio` transport and is launched by an MCP client. For a direct test:
```bash
python main.py
```
In this mode, without an MCP client, it won’t produce meaningful output; typical usage is via a client.

### Example Client (LangGraph + Gemini)
`try.py` launches the `main.py` MCP server over `stdio`, discovers tools, and wires them to a ReAct agent:
```bash
python try.py
```
The default sample sends a transcription request using files under `audio/`.

## Tools
Server name: `CustomAudioMCP`

All tools accept a list of file paths. Invalid paths or unsupported formats return an error. Supported formats: `.wav`, `.mp3`, `.ogg`, `.flac`.

### transcript
- Purpose: Converts audio files to text (Whisper base).
- Input:
  - `file_paths: List[str]`
  - `language: str = "en"`
  - `output_dir: str = "output"`
  - `overwrite: bool = False`
- Output:
  - `{ "transcripts": { "<path>": { "transcription": str, "transcript_file": str } | { "error": str } } }`
- Note: Processes at 16kHz and writes output under `output/transcripts/`.

### feature_analysis
- Purpose: Extracts basic features like pitch, tempo, duration; generates a waveform PNG.
- Input:
  - `file_paths: List[str]`
  - `output_dir: str = "output"`
  - `overwrite: bool = False`
- Output:
  - `{ "analyses": { "<path>": { "features": { "mean_pitch": float, "tempo": float, "duration_s": float }, "waveform_plot": str } | { "error": str } } }`
- Note: Waveform images are saved under `output/waveforms/`.

### audio_classification
- Purpose: Classifies audio content (AST model fine-tuned on AudioSet).
- Input:
  - `file_paths: List[str]`
  - `output_dir: str = "output"`
  - `output_csv: Optional[str] = None`
  - `overwrite: bool = False`
- Output:
  - `{ "classifications": { "<path>": { "label": str, "confidence": float } | { "error": str }, "csv_path"?: str, "csv_error"?: str } }`
- Note: Optionally writes a CSV into `output/classifications/`.

### metadata_extraction
- Purpose: Extracts metadata like duration, sample rate, channels, and tags; saves as JSON.
- Input:
  - `file_paths: List[str]`
  - `output_dir: str = "output"`
  - `overwrite: bool = False`
- Output:
  - `{ "metadata": { "<path>": { "metadata": { "duration_ms": int, "bitrate": int, "channels": int, "tags": object }, "metadata_file": str } | { "error": str } } }`
- Note: JSON files are saved under `output/metadata/`.

### audio_conversion
- Purpose: Converts files to the target audio format.
- Input:
  - `file_paths: List[str]`
  - `target_format: str = "wav"`
  - `output_dir: str = "output"`
  - `overwrite: bool = False`
- Output:
  - `{ "converted_files": { "<path>": { "converted_file": str } | { "error": str } } }`
- Note: Outputs are written under `output/converted/`.

## Example Usage
The following examples represent pseudo inputs that an MCP client would translate into tool calls:

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

## Outputs and Directory Structure
- `output/`
  - `transcripts/`: `*_transcript_<timestamp>.txt`
  - `waveforms/`: `*_waveform_<timestamp>.png`
  - `classifications/`: `labels.csv` (optional)
  - `metadata/`: `*_metadata_<timestamp>.json`
  - `converted/`: `*_converted_<timestamp>.<ext>`

## Notes and Tips
- If a GPU is available, `cuda` will be used automatically; otherwise `cpu`.
- First run may take longer due to model downloads.
- If a `config.json` file exists, it updates defaults like `output_dir`, `sample_rate`, and `overwrite_files`.
- Tools return error messages for unsupported formats or non-existent paths.



