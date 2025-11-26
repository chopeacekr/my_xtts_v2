# XTTS v2 TTS Server

Coqui XTTS v2 기반 HTTP TTS 서버 - 화자 복제(Voice Cloning) 기능 지원

## 🎯 주요 기능

- **🎤 화자 복제**: 사용자 음성 샘플로 목소리 복제
- **🌍 다국어 지원**: 14개 이상의 언어 지원 (한국어, 영어, 일본어, 중국어 등)
- **🔊 고품질 음성**: 자연스럽고 표현력 있는 음성 합성
- **⚡ FastAPI 기반**: RESTful API 제공
- **🎚️ 속도 조절**: 음성 속도 커스터마이징 (0.5x ~ 2.0x)

## 📋 시스템 요구사항

- **Python**: 3.11 이상
- **Package Manager**: UV
- **GPU**: CUDA 지원 GPU 권장 (CPU도 가능하지만 느림)
- **메모리**: 
  - GPU: 최소 4GB VRAM
  - CPU: 최소 8GB RAM

## 🚀 설치 방법

### 1. 프로젝트 클론
```bash
git clone <repository-url>
cd my_xtts_v2
```

### 2. UV를 사용한 의존성 설치
```bash
# 가상환경 생성 및 패키지 설치
uv sync
```

### 3. 주요 의존성
```toml
[project]
name = "my-xtts-v2"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "torch==2.3.1",              # PyTorch
    "torchaudio==2.3.1",         # 오디오 처리 (필수!)
    "coqui-tts==0.25.3",         # Coqui TTS 엔진
    "fastapi>=0.122.0",          # API 서버
    "uvicorn[standard]>=0.38.0", # ASGI 서버
    "soundfile>=0.13.1",         # 오디오 파일 I/O
    
    # 다국어 처리
    "jieba>=0.42.1",             # 중국어 토크나이저
    "cn2an>=0.5.23",             # 중국어 숫자 변환
    "pypinyin==0.50.0",          # 중국어 병음
    "fugashi>=1.5.2",            # 일본어 형태소 분석
    "cutlet>=0.5.0",             # 일본어 로마자 변환
    "unidic-lite>=1.0.8",        # 일본어 사전
    "hangul-romanize>=0.1.0",    # 한글 로마자 변환
]
```

## 🎮 실행 방법

### 기본 실행
```bash
cd my_xtts_v2
uv run uvicorn server_tts:app --host 0.0.0.0 --port 8100
```

서버가 시작되면:
```
============================================================
🚀 XTTS v2 Server Starting...
ℹ️  Device: cuda
============================================================
📦 Loading XTTS v2 model...
✅ Model loaded successfully in 12.34s
============================================================
✅ Server ready to synthesize speech!
============================================================
INFO:     Uvicorn running on http://0.0.0.0:8100
```

### 로깅 레벨 조정

`server_tts.py` 파일 상단:
```python
# 🎚️ 로깅 설정 (여기만 수정하세요!)
VERBOSE = True   # False: 최소 로그만
DEBUG = True     # False: 상세 정보 숨김
```

| 설정 | 용도 | 출력 |
|------|------|------|
| `VERBOSE=True, DEBUG=True` | 개발/디버깅 | 모든 상세 정보 |
| `VERBOSE=True, DEBUG=False` | 운영 | 핵심 로그만 |
| `VERBOSE=False, DEBUG=False` | 성능 테스트 | 최소 로그 |

## 📡 API 엔드포인트

### 1. Health Check

상태 확인 및 디바이스 정보 조회
```bash
GET http://localhost:8100/health
```

**응답 예시:**
```json
{
  "status": "ok",
  "device": "cuda"
}
```

### 2. TTS 합성 (Base64)

텍스트를 음성으로 변환하고 Base64로 반환
```bash
POST http://localhost:8100/synthesize_base64
Content-Type: application/json

{
  "text": "안녕하세요! 오늘 날씨가 정말 좋네요.",
  "lang": "ko",
  "speed": 1.0,
  "speaker_wav_b64": "<base64 encoded wav file>"
}
```

#### 요청 파라미터

| 파라미터 | 타입 | 필수 | 기본값 | 설명 |
|---------|------|------|--------|------|
| `text` | string | ✅ | - | 합성할 텍스트 (최대 5000자 권장) |
| `lang` | string | ❌ | `"ko"` | 언어 코드 (아래 참조) |
| `speed` | float | ❌ | `1.0` | 속도 (0.5~2.0) |
| `speaker_wav_b64` | string | ❌ | `null` | 화자 음성 샘플 (Base64 인코딩) |

#### 지원 언어 코드

| 코드 | 언어 | 코드 | 언어 |
|------|------|------|------|
| `ko` | 한국어 | `en` | 영어 |
| `ja` | 일본어 | `zh-cn` | 중국어 |
| `fr` | 프랑스어 | `de` | 독일어 |
| `es` | 스페인어 | `it` | 이탈리아어 |
| `pt` | 포르투갈어 | `pl` | 폴란드어 |
| `tr` | 터키어 | `ru` | 러시아어 |
| `nl` | 네덜란드어 | `cs` | 체코어 |

#### 응답 예시
```json
{
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
  "mime_type": "audio/wav"
}
```

## 💻 사용 예시

### Python 클라이언트

#### 기본 사용 (기본 화자)
```python
import requests
import base64

response = requests.post(
    "http://localhost:8100/synthesize_base64",
    json={
        "text": "안녕하세요! 테스트입니다.",
        "lang": "ko",
        "speed": 1.0
    },
    timeout=180
)

# 오디오 저장
audio_b64 = response.json()["audio_base64"]
audio_bytes = base64.b64decode(audio_b64)

with open("output.wav", "wb") as f:
    f.write(audio_bytes)

print("✅ 음성 파일 생성 완료: output.wav")
```

#### 화자 복제 사용
```python
import requests
import base64

# 1. 화자 음성 샘플 읽기
with open("my_voice.wav", "rb") as f:
    speaker_b64 = base64.b64encode(f.read()).decode()

# 2. TTS 요청 (화자 복제)
response = requests.post(
    "http://localhost:8100/synthesize_base64",
    json={
        "text": "이것은 제 목소리로 합성된 음성입니다.",
        "lang": "ko",
        "speed": 1.0,
        "speaker_wav_b64": speaker_b64  # 화자 음성 포함
    },
    timeout=180
)

# 3. 결과 저장
audio_b64 = response.json()["audio_base64"]
audio_bytes = base64.b64decode(audio_b64)

with open("cloned_voice.wav", "wb") as f:
    f.write(audio_bytes)

print("✅ 복제된 음성 생성 완료: cloned_voice.wav")
```

### cURL 예시

#### 기본 화자
```bash
curl -X POST http://localhost:8100/synthesize_base64 \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test.",
    "lang": "en",
    "speed": 1.0
  }' \
  --max-time 180
```

#### Health Check
```bash
curl http://localhost:8100/health
```

## 📁 프로젝트 구조
```
my_xtts_v2/
├── server_tts.py           # FastAPI TTS 서버
├── pyproject.toml          # 프로젝트 의존성 및 메타데이터
├── README.md               # 이 문서
└── .venv/                  # 가상환경 (자동 생성)
```

## ⚙️ 성능 최적화

### GPU vs CPU
```python
# server_tts.py
device = "cuda" if torch.cuda.is_available() else "cpu"
```

| 환경 | 첫 요청 | 이후 요청 | 권장 사용 |
|------|---------|-----------|-----------|
| **CUDA (GPU)** | 8~15초 | 5~10초 | ✅ 권장 |
| **CPU** | 30~60초 | 20~40초 | ⚠️ 느림 |

### 처리 시간 구성

로그에서 확인 가능 (`DEBUG=True`):
```
✅ Request completed (8.80s)
   Breakdown:
     Speaker: 0.12s      # 화자 임베딩 생성
     Synthesis: 8.45s    # TTS 합성 (가장 오래 걸림)
     Encoding: 0.23s     # Base64 인코딩
```

### 메모리 요구사항

- **GPU 모드**: 
  - VRAM: 2~4GB (기본 화자)
  - VRAM: 4~6GB (화자 복제)
  
- **CPU 모드**:
  - RAM: 4~8GB

## 🐛 문제 해결

### 1. `torchcodec` 에러
```bash
ImportError: TorchCodec is required for load_with_torchcodec
```

**원인**: `torchaudio 2.9+`가 torchcodec을 요구함

**해결책**:
```bash
cd my_xtts_v2
uv pip uninstall torchaudio
uv pip install torchaudio==2.3.1
```

또는 `pyproject.toml`을 확인:
```toml
dependencies = [
    "torchaudio==2.3.1",  # 버전 고정 필수!
]
```

### 2. 타임아웃 에러
```
ReadTimeout: Read timed out. (read timeout=60)
```

**원인**: 첫 요청은 화자 임베딩 생성으로 시간이 오래 걸림

**해결책**: 클라이언트 타임아웃 증가
```python
response = requests.post(..., timeout=180)  # 60초 → 180초
```

**참고**: 
- 첫 요청: 30~60초 (GPU), 60~120초 (CPU)
- 이후 요청: 5~10초 (GPU), 20~40초 (CPU)

### 3. GPU 메모리 부족
```
CUDA out of memory
```

**해결책 1**: CPU로 전환
```python
# server_tts.py
device = "cpu"
```

**해결책 2**: 다른 GPU 프로세스 종료
```bash
nvidia-smi  # GPU 사용 현황 확인
kill <PID>  # 불필요한 프로세스 종료
```

### 4. 일본어 Tokenizer 문제

**증상**: 일본어 입력 시 에러 발생

**임시 우회책**: `lang="en"` 사용
```python
# 일본어 텍스트지만 lang="en" 사용
response = requests.post(..., json={
    "text": "こんにちは、世界",
    "lang": "en"  # "ja" 대신 "en"
})
```

### 5. MeCab 에러
```
MeCab dictionary is not found
```

**해결책**: MeCab 사전 설치
```bash
# Ubuntu/Debian
sudo apt-get install mecab mecab-ipadic-utf8

# macOS
brew install mecab mecab-ipadic
```

## 🔧 고급 설정

### 1. 포트 변경
```bash
uv run uvicorn server_tts:app --host 0.0.0.0 --port 9000
```

### 2. HTTPS 활성화
```bash
uv run uvicorn server_tts:app \
  --host 0.0.0.0 \
  --port 8100 \
  --ssl-keyfile=/path/to/key.pem \
  --ssl-certfile=/path/to/cert.pem
```

### 3. 워커 수 증가 (병렬 처리)
```bash
uv run uvicorn server_tts:app --workers 4
```

⚠️ **주의**: 워커당 별도 GPU 메모리 필요 (4GB × 워커 수)

### 4. 로그 파일 저장
```bash
uv run uvicorn server_tts:app \
  --log-config logging.yaml \
  > server.log 2>&1
```

## 📊 성능 벤치마크

테스트 환경: NVIDIA RTX 3090, AMD Ryzen 9 5900X

| 텍스트 길이 | GPU (첫 요청) | GPU (이후) | CPU |
|------------|--------------|-----------|-----|
| 짧음 (10자) | 8.2초 | 4.5초 | 25초 |
| 보통 (50자) | 10.5초 | 6.8초 | 35초 |
| 긴 글 (200자) | 15.3초 | 12.1초 | 75초 |

## 🆚 다른 TTS 비교

| 항목 | XTTS v2 | MeloTTS | Google Cloud TTS |
|------|---------|---------|------------------|
| **화자 복제** | ✅ 가능 | ❌ 불가 | ❌ 불가 |
| **속도** | 🐢 느림 (8~10초) | 🚀 빠름 (1~2초) | ⚡ 매우 빠름 (<1초) |
| **음질** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **자연스러움** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **GPU 필요** | ✅ 권장 | ❌ 불필요 | ❌ 클라우드 |
| **비용** | 🆓 무료 (로컬) | 🆓 무료 (로컬) | 💰 종량제 |
| **오프라인** | ✅ 가능 | ✅ 가능 | ❌ 불가 |
| **상업적 이용** | ✅ MPL 2.0 | ✅ MIT | ⚠️ 약관 확인 |

## 📝 라이선스

- **프로젝트**: MIT License
- **Coqui TTS**: Mozilla Public License 2.0
- **의존 라이브러리**: 각 라이브러리의 라이선스 참조

## 🤝 기여

이슈 제보 및 풀 리퀘스트를 환영합니다!

### 기여 방법
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 참고 자료

- [Coqui TTS GitHub](https://github.com/coqui-ai/TTS)
- [XTTS v2 Paper](https://arxiv.org/abs/2406.04904)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [UV Documentation](https://github.com/astral-sh/uv)

## 🙋 FAQ

**Q: 화자 복제에 필요한 음성 샘플 길이는?**  
A: 최소 6초, 권장 10~30초. 깨끗하고 명확한 음성일수록 좋습니다.

**Q: 여러 언어를 동시에 합성할 수 있나요?**  
A: 네, 언어별로 별도 요청하면 됩니다. 모델은 한 번만 로드됩니다.

**Q: 실시간 스트리밍이 가능한가요?**  
A: 현재 버전은 파일 기반입니다. 스트리밍은 추후 업데이트 예정입니다.

**Q: 상업적으로 사용할 수 있나요?**  
A: Coqui TTS는 MPL 2.0 라이선스로 상업적 사용 가능합니다.

## 📧 문의

- **이슈 제보**: [GitHub Issues](링크)
- **이메일**: chopeacekr@gmail.com
---

**Version**: 0.1.0  
**Last Updated**: 2024-11-26  
**Made with** ❤️ **by Peace Cho**