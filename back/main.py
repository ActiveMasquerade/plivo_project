import os
import io
import tempfile
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# image captioning imports
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch

# audio/diarization imports
import numpy as np
import librosa
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import KMeans
from pydub import AudioSegment
import whisper

# Fix HF cache permission if containerized
os.environ["HF_HOME"] = "/tmp/huggingface"

app = FastAPI(title="Multimodal Playground API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Image captioning: smaller CPU-friendly model
# -----------------------
# (kept minimal — your existing code already uses this)
img_model_name = "nlpconnect/vit-gpt2-image-captioning"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading image captioning model...")
img_model = VisionEncoderDecoderModel.from_pretrained(img_model_name).to(device)
feature_extractor = ViTImageProcessor.from_pretrained(img_model_name)
tokenizer = AutoTokenizer.from_pretrained(img_model_name)
print("Image model loaded.")

@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
        output_ids = img_model.generate(pixel_values, max_length=32, num_beams=4)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return {"caption": caption}
    except Exception as e:
        return {"error": str(e)}

# -----------------------
# Conversation Analysis: diarization + STT
# -----------------------

# Load Whisper model for STT (choose 'tiny','base','small','medium','large' depending on resources)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")  # change to 'tiny' for fastest CPU
print(f"Loading Whisper ASR model '{WHISPER_MODEL}' (this may take a while)...")
whisper_model = whisper.load_model(WHISPER_MODEL, device=str(device))
print("Whisper loaded.")

# Resemblyzer (for speaker embeddings)
print("Initializing resemblyzer voice encoder...")
voice_encoder = VoiceEncoder()
print("Voice encoder ready.")

class Segment(BaseModel):
    start: float
    end: float
    speaker: int

def vad_segments(y: np.ndarray, sr: int, hop_length: int = 512, energy_threshold_ratio: float = 0.5):
    """Simple energy-based VAD returning list of (start, end) in seconds."""
    energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(energy)), sr=sr, hop_length=hop_length)
    # boolean voiced frames
    thresh = np.median(energy) * energy_threshold_ratio
    voiced = energy > thresh
    segments = []
    if not voiced.any():
        return segments
    # merge contiguous voiced frames
    start_idx = None
    for i, v in enumerate(voiced):
        if v and start_idx is None:
            start_idx = i
        if not v and start_idx is not None:
            s = times[start_idx]
            e = times[i]
            segments.append((s, e))
            start_idx = None
    if start_idx is not None:
        segments.append((times[start_idx], times[-1]))
    # filter very short segments
    segments = [(max(0, s-0.02), e+0.02) for s,e in segments if (e - s) > 0.2]
    return segments

def compute_segment_embeddings(audio_path: str, segments: List[tuple], sr=16000):
    """Given audio and segments, return list of embeddings and segment times."""
    emb_list = []
    seg_times = []
    # load as wave
    wav, sr_loaded = librosa.load(audio_path, sr=sr)
    for (s,e) in segments:
        start_sample = int(s * sr)
        end_sample = int(e * sr)
        chunk = wav[start_sample:end_sample]
        if len(chunk) < 100:  # too short
            continue
        # write temp wav for resemblyzer preprocess
        tmpf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmpf.name, chunk, sr)
        try:
            wav_proc = preprocess_wav(tmpf.name)
            emb = voice_encoder.embed_utterance(wav_proc)
            emb_list.append(emb)
            seg_times.append((s,e))
        finally:
            try:
                os.unlink(tmpf.name)
            except:
                pass
    return emb_list, seg_times

def cluster_embeddings(embs: np.ndarray, max_speakers: int = 2):
    k = min(max_speakers, len(embs))
    if k <= 1:
        return [0] * len(embs)
    km = KMeans(n_clusters=k, random_state=0).fit(np.vstack(embs))
    return km.labels_

def split_audio_to_file(orig_path: str, start: float, end: float, out_path: str):
    # pydub works in ms
    audio = AudioSegment.from_file(orig_path)
    seg = audio[start*1000:end*1000]
    seg.export(out_path, format="wav")

@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...), max_speakers: int = 2):
    """
    Upload an audio file. Returns diarization (segments -> speaker ids) and transcripts per segment.
    max_speakers: up to 2 (but default param accepts more; logic clusters to <=2)
    """
    # Save uploaded audio to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix="." + (file.filename.split(".")[-1] if "." in file.filename else "wav"))
    contents = await file.read()
    tmp.write(contents)
    tmp.flush()
    tmp.close()
    audio_path = tmp.name

    try:
        # Ensure consistent sr for VAD
        y, sr = librosa.load(audio_path, sr=16000)
        # 1) VAD -> rough speech segments
        raw_segments = vad_segments(y, sr)
        if len(raw_segments) == 0:
            return {"error": "No speech detected."}

        # 2) embed each segment
        embs, seg_times = compute_segment_embeddings(audio_path, raw_segments, sr=16000)
        if len(embs) == 0:
            return {"error": "No valid segments for embedding."}

        # 3) cluster into up to max_speakers groups
        labels = cluster_embeddings(embs, max_speakers=max_speakers)

        # 4) build diarization segments with speaker labels (map labels to 0/1)
        diar_segments = []
        for (s,e), lab in zip(seg_times, labels):
            diar_segments.append({"start": float(s), "end": float(e), "speaker": int(lab)})

        # 5) For each diarized segment, extract audio chunk and run STT (Whisper)
        transcripts = []
        for i, seg in enumerate(diar_segments):
            seg_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            split_audio_to_file(audio_path, seg["start"], seg["end"], seg_file.name)
            # Whisper transcription
            # choose whisper options: language=None auto-detect, task="transcribe"
            res = whisper_model.transcribe(seg_file.name, language=None, task="transcribe")
            text = res.get("text", "").strip()
            transcripts.append({
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
                "whisper_raw": {"language": res.get("language"), "segments": res.get("segments")}
            })
            try:
                os.unlink(seg_file.name)
            except:
                pass

        # 6) Optionally merge contiguous segments of same speaker into a single transcript chunk
        merged = []
        for t in transcripts:
            if not merged:
                merged.append(t.copy())
            else:
                last = merged[-1]
                if last["speaker"] == t["speaker"] and abs(t["start"] - last["end"]) < 0.5:
                    # merge
                    last["end"] = t["end"]
                    last["text"] = (last["text"] + " " + t["text"]).strip()
                else:
                    merged.append(t.copy())

        return {
            "diarization": diar_segments,
            "transcripts": transcripts,
            "merged_transcripts": merged
        }

    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            os.unlink(audio_path)
        except:
            pass
