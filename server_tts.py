import os
import base64
from tempfile import NamedTemporaryFile
import time
import torch
from TTS.api import TTS
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ================================
# 🎚️ 로깅 설정 (여기만 수정하세요!)
# ================================
VERBOSE = True  # False로 변경하면 최소 로그만 출력
DEBUG = True    # False로 변경하면 상세 정보 숨김

# 일본어 MeCab 설정
os.environ.setdefault("MECABRC", "/var/lib/mecab/dic/debian/mecabrc")

device = "cuda" if torch.cuda.is_available() else "cpu"

def log(message: str, level: str = "INFO"):
    """로깅 함수"""
    if not VERBOSE:
        return
    
    emoji_map = {
        "INFO": "ℹ️ ",
        "SUCCESS": "✅",
        "ERROR": "❌",
        "WARNING": "⚠️ ",
        "START": "🚀",
        "REQUEST": "📨",
        "PROCESS": "🔊",
        "FILE": "📁",
        "CLEAN": "🗑️ ",
        "TIME": "⏱️ ",
    }
    
    emoji = emoji_map.get(level, "  ")
    print(f"{emoji} {message}")

def log_debug(message: str):
    """디버그 로깅 (DEBUG=True일 때만 출력)"""
    if DEBUG and VERBOSE:
        print(f"   {message}")

def log_separator(char="=", length=60):
    """구분선 출력"""
    if VERBOSE:
        print(char * length)

# 서버 시작
log_separator()
log("XTTS v2 Server Starting...", "START")
log(f"Device: {device}", "INFO")
log_separator()

_original_torch_load = torch.load

def _patched_torch_load(f, map_location=None, **kwargs):
    if map_location is None:
        map_location = device
    return _original_torch_load(f, map_location=map_location, **kwargs)

def load_xtts_model():
    model_id = "tts_models/multilingual/multi-dataset/xtts_v2"
    log(f"Loading XTTS v2 model...", "INFO")
    log_debug(f"Model ID: {model_id}")
    start_time = time.time()
    
    torch.load = _patched_torch_load
    try:
        model = TTS(model_id).to(device)
        elapsed = time.time() - start_time
        log(f"Model loaded successfully in {elapsed:.2f}s", "SUCCESS")
    finally:
        torch.load = _original_torch_load
    return model

xtts_model = load_xtts_model()
log_separator()
log("Server ready to synthesize speech!", "SUCCESS")
log_separator()

app = FastAPI(title="XTTS v2 TTS Server")

class TTSRequest(BaseModel):
    text: str
    lang: str = "ko"
    speed: float = 1.0
    speaker_wav_b64: str | None = None

class TTSResponse(BaseModel):
    audio_base64: str
    mime_type: str = "audio/wav"

@app.get("/health")
def health():
    log("Health check requested", "INFO")
    return {"status": "ok", "device": device}

@app.post("/synthesize_base64", response_model=TTSResponse)
def synthesize_base64(req: TTSRequest):
    request_start = time.time()
    
    if VERBOSE:
        print()  # 빈 줄
        log_separator()
    
    log("New TTS Request", "REQUEST")
    log_debug(f"Language: {req.lang}")
    log_debug(f"Speed: {req.speed}")
    log_debug(f"Text length: {len(req.text)} chars")
    log_debug(f"Has speaker reference: {bool(req.speaker_wav_b64)}")
    if DEBUG:
        preview = req.text[:100] + ("..." if len(req.text) > 100 else "")
        log_debug(f"Text preview: {preview}")
    
    text = req.text.strip()
    if not text:
        log("Text is empty", "ERROR")
        raise HTTPException(status_code=400, detail="Text is empty")
    
    with NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
        out_path = tmp_out.name
    log_debug(f"Temp output: {out_path}")
    
    speaker_wav_path = None
    tmp_speaker_file = None
    speaker_elapsed = 0
    
    try:
        # Speaker reference 처리
        if req.speaker_wav_b64:
            speaker_start = time.time()
            log("Processing speaker reference...", "PROCESS")
            from base64 import b64decode
            
            speaker_data = b64decode(req.speaker_wav_b64)
            log_debug(f"Speaker audio size: {len(speaker_data)} bytes")
            
            with NamedTemporaryFile(suffix=".wav", delete=False) as tmp_spk:
                tmp_speaker_file = tmp_spk.name
                tmp_spk.write(speaker_data)
            speaker_wav_path = tmp_speaker_file
            
            speaker_elapsed = time.time() - speaker_start
            log(f"Speaker reference ready ({speaker_elapsed:.2f}s)", "SUCCESS")
            log_debug(f"Saved to: {tmp_speaker_file}")
        else:
            log_debug("Using default speaker")
        
        # TTS 생성
        tts_start = time.time()
        log("Synthesizing speech...", "PROCESS")
        log_debug(f"Mode: {'Voice cloning' if speaker_wav_path else 'Default speaker'}")
        
        if speaker_wav_path:
            xtts_model.tts_to_file(
                text=text,
                file_path=out_path,
                speaker_wav=speaker_wav_path,
                language=req.lang,
                speed=req.speed,
            )
        else:
            xtts_model.tts_to_file(
                text=text,
                file_path=out_path,
                language=req.lang,
                speed=req.speed,
            )
        
        tts_elapsed = time.time() - tts_start
        log(f"Synthesis completed ({tts_elapsed:.2f}s)", "SUCCESS")
        
        # 파일 읽기 및 base64 인코딩
        encode_start = time.time()
        log_debug("Encoding to base64...")
        
        with open(out_path, "rb") as f:
            audio_bytes = f.read()
        
        if DEBUG:
            file_size_kb = len(audio_bytes) / 1024
            log_debug(f"Audio size: {file_size_kb:.2f} KB")
        
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        encode_elapsed = time.time() - encode_start
        log_debug(f"Encoded in {encode_elapsed:.2f}s")
        
        # 총 처리 시간
        total_elapsed = time.time() - request_start
        log(f"Request completed ({total_elapsed:.2f}s)", "SUCCESS")
        
        if DEBUG:
            log_debug(f"Breakdown:")
            if speaker_elapsed > 0:
                log_debug(f"  Speaker: {speaker_elapsed:.2f}s")
            log_debug(f"  Synthesis: {tts_elapsed:.2f}s")
            log_debug(f"  Encoding: {encode_elapsed:.2f}s")
        
        if VERBOSE:
            log_separator()
        
        return TTSResponse(audio_base64=audio_b64, mime_type="audio/wav")
        
    except Exception as e:
        error_elapsed = time.time() - request_start
        log(f"ERROR after {error_elapsed:.2f}s: {str(e)}", "ERROR")
        
        if DEBUG:
            import traceback
            print("\n📋 Full traceback:")
            traceback.print_exc()
        
        if VERBOSE:
            log_separator()
        
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {e}") from e
        
    finally:
        # Cleanup
        cleaned_files = []
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
                cleaned_files.append("output")
        except Exception as e:
            log_debug(f"Failed to remove output: {e}")
        
        if tmp_speaker_file:
            try:
                if os.path.exists(tmp_speaker_file):
                    os.remove(tmp_speaker_file)
                    cleaned_files.append("speaker")
            except Exception as e:
                log_debug(f"Failed to remove speaker: {e}")
        
        if DEBUG and cleaned_files:
            log_debug(f"Cleaned: {', '.join(cleaned_files)}")