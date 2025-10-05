from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any
import os
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from mutagen import File as MutagenFile
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa
import asyncio
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import json

# Desteklenen ses formatları
SUPPORTED_FORMATS = {".wav", ".mp3", ".ogg", ".flac"}

# Varsayılan konfigürasyon
DEFAULT_CONFIG = {
    "output_dir": "output",
    "sample_rate": 16000,
    "overwrite_files": False
}

# Modelleri preload et (hafıza verimli)
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)
feature_extractor = pipeline("feature-extraction", model="facebook/wav2vec2-base")
classifier = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")

# FastMCP instance'ı oluştur
mcp = FastMCP("CustomAudioMCP")

def validate_file(path: str) -> bool:
    """Dosyanın varlığını ve formatını doğrula."""
    return os.path.exists(path) and any(path.lower().endswith(fmt) for fmt in SUPPORTED_FORMATS)

def get_timestamp() -> str:
    """Zaman damgası oluştur."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

@mcp.tool()
async def transcript(file_paths: List[str], language: str = "en", output_dir: str = DEFAULT_CONFIG["output_dir"], overwrite: bool = DEFAULT_CONFIG["overwrite_files"]) -> Dict[str, Any]:
    """Ses dosyalarını transkribe eder.

    Args:
        file_paths: Transkribe edilecek ses dosyalarının yolları (liste).
        language: Dil kodu (varsayılan 'en').
        output_dir: Çıktı klasörü (varsayılan 'output').
        overwrite: Var olan dosyaların üzerine yaz (varsayılan False).

    Returns:
        Dict[str, Any]: Her ses dosyası için transkripsiyonlar ve hata mesajları.
    """
    transcript_dir = os.path.join(output_dir, "transcripts")
    os.makedirs(transcript_dir, exist_ok=True)
    results = {}
    
    for path in tqdm(file_paths, desc="Transkripsiyon yapılıyor"):
        if not validate_file(path):
            results[path] = {"error": "Dosya bulunamadı veya desteklenmeyen format"}
            continue
        try:
            audio, sr = librosa.load(path, sr=DEFAULT_CONFIG["sample_rate"])
            inputs = whisper_processor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(device)
            with torch.no_grad():
                predicted_ids = whisper_model.generate(inputs)
            transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            # Transkripsiyonu metin dosyasına kaydet
            transcript_path = os.path.join(transcript_dir, f"{os.path.basename(path).rsplit('.', 1)[0]}_transcript_{get_timestamp()}.txt")
            if not os.path.exists(transcript_path) or overwrite:
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(transcription)
            results[path] = {"transcription": transcription, "transcript_file": transcript_path}
        except Exception as e:
            results[path] = {"error": f"Transkripsiyon hatası: {str(e)}"}
    
    return {"transcripts": results}

@mcp.tool()
def feature_analysis(file_paths: List[str], output_dir: str = DEFAULT_CONFIG["output_dir"], overwrite: bool = DEFAULT_CONFIG["overwrite_files"]) -> Dict[str, Any]:
    """Ses özelliklerini analiz eder (pitch, tempo vb.) ve dalga formunu PNG olarak kaydeder.

    Args:
        file_paths: Analiz edilecek ses dosyalarının yolları (liste).
        output_dir: Çıktı klasörü (varsayılan 'output').
        overwrite: Var olan dosyaların üzerine yaz (varsayılan False).

    Returns:
        Dict[str, Any]: Her ses dosyası için özellikler ve dalga formu dosya yolları.
    """
    waveform_dir = os.path.join(output_dir, "waveforms")
    os.makedirs(waveform_dir, exist_ok=True)
    results = {}
    
    for path in tqdm(file_paths, desc="Özellik analizi yapılıyor"):
        if not validate_file(path):
            results[path] = {"error": "Dosya bulunamadı veya desteklenmeyen format"}
            continue
        try:
            audio, sr = librosa.load(path, sr=DEFAULT_CONFIG["sample_rate"])
            features = {
                "mean_pitch": float(np.mean(librosa.yin(audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')))),
                "tempo": float(librosa.beat.tempo(y=audio, sr=sr)[0]),
                "duration_s": float(len(audio) / sr)
            }
            # Dalga formu grafiği oluştur ve kaydet
            plot_path = os.path.join(waveform_dir, f"{os.path.basename(path).rsplit('.', 1)[0]}_waveform_{get_timestamp()}.png")
            if not os.path.exists(plot_path) or overwrite:
                fig, ax = plt.subplots()
                ax.plot(audio)
                ax.set_title(f"Dalga Formu: {os.path.basename(path)}")
                ax.set_xlabel("Zaman (örnek)")
                ax.set_ylabel("Genlik")
                fig.savefig(plot_path, format="png", dpi=300)
                plt.close(fig)
            results[path] = {"features": features, "waveform_plot": plot_path}
        except Exception as e:
            results[path] = {"error": f"Analiz hatası: {str(e)}"}
    
    return {"analyses": results}

@mcp.tool()
def audio_classification(file_paths: List[str], output_dir: str = DEFAULT_CONFIG["output_dir"], output_csv: str = None, overwrite: bool = DEFAULT_CONFIG["overwrite_files"]) -> Dict[str, Any]:
    """Ses dosyalarını sınıflandırır (müzik, konuşma vb.) ve sonuçları opsiyonel olarak CSV dosyasına kaydeder.

    Args:
        file_paths: Sınıflandırılacak ses dosyalarının yolları (liste).
        output_dir: Çıktı klasörü (varsayılan 'output').
        output_csv: Etiketlerin kaydedileceği CSV dosyasının adı (opsiyonel, örneğin 'labels.csv').
        overwrite: Var olan CSV dosyasının üzerine yaz (varsayılan False).

    Returns:
        Dict[str, Any]: Her ses dosyası için sınıflandırma etiketleri ve güven skorları.
    """
    classification_dir = os.path.join(output_dir, "classifications")
    os.makedirs(classification_dir, exist_ok=True)
    results = {}
    csv_rows = []
    
    for path in tqdm(file_paths, desc="Sınıflandırma yapılıyor"):
        if not validate_file(path):
            results[path] = {"error": "Dosya bulunamadı veya desteklenmeyen format"}
            continue
        try:
            with torch.no_grad():
                classification = classifier(path)
            label = classification[0]["label"]
            score = float(classification[0]["score"])
            results[path] = {"label": label, "confidence": score}
            if output_csv:
                csv_rows.append({
                    "file": path,
                    "label": label,
                    "confidence": score,
                    "timestamp": get_timestamp()
                })
        except Exception as e:
            results[path] = {"error": f"Sınıflandırma hatası: {str(e)}"}
    
    if output_csv and csv_rows:
        try:
            csv_path = os.path.join(classification_dir, output_csv)
            if not os.path.exists(csv_path) or overwrite:
                pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
                results["csv_path"] = csv_path
            else:
                results["csv_error"] = "CSV dosyası zaten var ve üzerine yazma izni yok"
        except Exception as e:
            results["csv_error"] = f"CSV kaydetme hatası: {str(e)}"
    
    return {"classifications": results}

@mcp.tool()
def metadata_extraction(file_paths: List[str], output_dir: str = DEFAULT_CONFIG["output_dir"], overwrite: bool = DEFAULT_CONFIG["overwrite_files"]) -> Dict[str, Any]:
    """Ses metadata'sını çıkarır (süre, bitrate vb.) ve JSON olarak kaydeder.

    Args:
        file_paths: Metadata çıkarılacak ses dosyalarının yolları (liste).
        output_dir: Çıktı klasörü (varsayılan 'output').
        overwrite: Var olan dosyaların üzerine yaz (varsayılan False).

    Returns:
        Dict[str, Any]: Her ses dosyası için metadata ve JSON dosya yolları.
    """
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    results = {}
    
    for path in tqdm(file_paths, desc="Metadata çıkarılıyor"):
        if not validate_file(path):
            results[path] = {"error": "Dosya bulunamadı veya desteklenmeyen format"}
            continue
        try:
            audio = AudioSegment.from_file(path)
            meta = MutagenFile(path)
            metadata = {
                "duration_ms": len(audio),
                "bitrate": audio.frame_rate,
                "channels": audio.channels,
                "tags": dict(meta.tags) if meta and meta.tags else {}
            }
            # Metadata'yı JSON olarak kaydet
            metadata_path = os.path.join(metadata_dir, f"{os.path.basename(path).rsplit('.', 1)[0]}_metadata_{get_timestamp()}.json")
            if not os.path.exists(metadata_path) or overwrite:
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            results[path] = {"metadata": metadata, "metadata_file": metadata_path}
        except Exception as e:
            results[path] = {"error": f"Metadata çıkarma hatası: {str(e)}"}
    
    return {"metadata": results}

@mcp.tool()
def audio_conversion(file_paths: List[str], target_format: str = "wav", output_dir: str = DEFAULT_CONFIG["output_dir"], overwrite: bool = DEFAULT_CONFIG["overwrite_files"]) -> Dict[str, Any]:
    """Ses formatlarını dönüştürür (.mp3 to .wav vb.).

    Args:
        file_paths: Dönüştürülecek ses dosyalarının yolları (liste).
        target_format: Hedef format (varsayılan 'wav').
        output_dir: Çıktı klasörü (varsayılan 'output').
        overwrite: Var olan dosyaların üzerine yaz (varsayılan False).

    Returns:
        Dict[str, Any]: Dönüştürülen ses dosyalarının yolları.
    """
    conversion_dir = os.path.join(output_dir, "converted")
    os.makedirs(conversion_dir, exist_ok=True)
    results = {}
    
    for path in tqdm(file_paths, desc="Ses dönüştürülüyor"):
        if not validate_file(path):
            results[path] = {"error": "Dosya bulunamadı veya desteklenmeyen format"}
            continue
        try:
            audio = AudioSegment.from_file(path)
            new_path = os.path.join(conversion_dir, f"{os.path.basename(path).rsplit('.', 1)[0]}_converted_{get_timestamp()}.{target_format}")
            if not os.path.exists(new_path) or overwrite:
                audio.export(new_path, format=target_format)
            results[path] = {"converted_file": new_path}
        except Exception as e:
            results[path] = {"error": f"Dönüştürme hatası: {str(e)}"}
    
    return {"converted_files": results}

if __name__ == "__main__":
    # Konfigürasyon dosyasını yükle (varsa)
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            DEFAULT_CONFIG.update(json.load(f))
    
    mcp.run(transport='stdio')