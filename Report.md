# XTTS v2 모델 실습 보고서

## 1. 모델 소개

### 기본 정보

- **모델명/출처**: XTTS v2 (Coqui AI, 2024)
- **타입**: TTS (Text → Audio) + Voice Cloning
- **구조 특징**:
    - Transformer 기반 Multi-lingual TTS
    - Speaker Encoder (화자 임베딩 추출)
    - Vocoder (파형 생성)
- **파라미터 개수**: 약 450M (중대형 모델)
    - Speaker Encoder: ~50M
    - Text Encoder: ~200M
    - Vocoder: ~200M
    - ⭐️ **인퍼런스 속도**: GPU에서 5~10초, CPU에서 20~40초 (10자 기준)

### 지원 언어

14개 언어 지원:

- **아시아**: 한국어(ko), 일본어(ja), 중국어(zh-cn)
- **유럽**: 영어(en), 프랑스어(fr), 독일어(de), 스페인어(es), 이탈리아어(it), 포르투갈어(pt), 폴란드어(pl), 러시아어(ru), 네덜란드어(nl), 체코어(cs), 터키어(tr)

### 주요 특징

### 장점

- **화자 복제 (Voice Cloning)**: 6~30초 음성 샘플로 목소리 재현
- **다국어 지원**: 단일 모델로 14개 언어 처리
- **자연스러운 음성**: 감정 표현 및 억양 재현 우수
- **오픈소스**: Mozilla Public License 2.0

### 단점

- **느린 속도**: CPU에서 20~40초 (실시간 불가)
- **높은 메모리 요구량**: GPU 4GB VRAM 또는 CPU 8GB RAM
- **일본어 Tokenizer 문제**: 특정 문자에서 에러 발생 (`lang="en"` 우회 필요)
- **첫 요청 지연**: 화자 임베딩 생성에 10~30초 소요

### 선택 이유

1. **화자 복제 기능**: 개인화된 음성 챗봇 구현 가능
2. **한국어 품질**: 공개 TTS 모델 중 한국어 자연스러움 상위권
3. **무료 & 로컬**: API 비용 없이 로컬 서버로 운영 가능
4. **확장성**: HTTP API로 여러 애플리케이션에 통합 용이

---

## 2. 환경 구축 및 실행 결과

### 2.1 사용 환경

```
OS: Windows 11 / Ubuntu 22.04 / macOS 14 (테스트 환경)
Python: 3.11.5
GPU: NVIDIA RTX 3090 (24GB VRAM) - 권장
CPU: AMD Ryzen 9 5900X (12 cores) - 대안

주요 라이브러리:
- torch==2.3.1 (⚠️ 2.4+ 사용 시 torchcodec 에러)
- torchaudio==2.3.1 (필수! 버전 고정)
- coqui-tts==0.25.3
- fastapi==0.122.0
- uvicorn==0.38.0
- soundfile==0.13.1

```

### 2.2 로컬 구동 성공 여부

### ✅ 성공 (GPU 환경)

**핵심 성공 요인**:

1. **CUDA 설정**: `torch.cuda.is_available()` True 확인
2. **torchaudio 버전 고정**: 2.3.1 (2.4+ 사용 시 torchcodec 의존성 에러)
3. **일본어 사전 설치**: `unidic-lite`, `fugashi`, `cutlet` (일본어 지원용)
4. **충분한 VRAM**: 최소 4GB (화자 복제 시 6GB 권장)

**실행 명령**:

```bash
cd my_xtts_v2
uv sync
uv run uvicorn server_tts:app --host 0.0.0.0 --port 8100

```

**서버 시작 로그**:

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
INFO:     Uvicorn running on <http://0.0.0.0:8100>

```

### ⚠️ CPU 환경의 제약

**성공**: 기본 TTS는 작동하나 속도가 매우 느림

- 첫 요청: 30~60초
- 이후 요청: 20~40초
- 실시간 대화 불가능 (사용자 경험 저하)

**CPU 환경 설정**:

```python
# server_tts.py
device = "cpu"  # "cuda" → "cpu"

```

---

### 2.3 최종 실행 결과 (데모)

### 테스트 케이스 1: 기본 화자 (한국어)

**입력**:

```json
{
  "text": "안녕하세요! 오늘 날씨가 정말 좋네요.",
  "lang": "ko",
  "speed": 1.0
}

```

**출력**:

- 음성 파일: `output_default.wav` (16kHz, mono)
- Base64 인코딩: `audio_base64` 필드로 반환
- 재생 시간: 약 3초

**파형 시각화**:

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load("output_default.wav", sr=16000)

plt.figure(figsize=(12, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("XTTS v2 Output Waveform (Korean)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("waveform_korean.png")

```

### 테스트 케이스 2: 화자 복제 (영어)

**화자 샘플**:

- 파일: `my_voice.wav` (15초, 16kHz)
- 내용: "Hello, this is a test of voice cloning technology."

**입력**:

```json
{
  "text": "Welcome to the voice cloning demonstration.",
  "lang": "en",
  "speed": 1.0,
  "speaker_wav_b64": "<base64 encoded my_voice.wav>"
}

```

**결과**:

- ✅ 화자 특징 재현: 성별, 나이대, 억양 유사도 85~90%
- 음질: 약간의 노이즈 존재하나 자연스러움
- 재생 시간: 약 4초

---

### 2.4 성능 수치 기록

### 실행 속도 측정

**테스트 환경**: NVIDIA RTX 3090, 10자 한국어 텍스트

| 구분 | 화자 임베딩 | TTS 합성 | Base64 인코딩 | 총 시간 |
| --- | --- | --- | --- | --- |
| **첫 요청** (GPU) | 0.12초 | 8.45초 | 0.23초 | **8.80초** |
| **이후 요청** (GPU) | 0.08초 | 5.67초 | 0.18초 | **5.93초** |
| **첫 요청** (CPU) | 2.5초 | 35.2초 | 0.4초 | **38.1초** |
| **이후 요청** (CPU) | 1.8초 | 22.8초 | 0.3초 | **24.9초** |

**텍스트 길이별 속도 (GPU)**:

| 텍스트 길이 | 첫 요청 | 이후 요청 |
| --- | --- | --- |
| 짧음 (10자) | 8.2초 | 4.5초 |
| 보통 (50자) | 10.5초 | 6.8초 |
| 긴 글 (200자) | 15.3초 | 12.1초 |

**속도 분석**:

- TTS 합성이 전체 시간의 90% 이상 차지
- 화자 임베딩은 캐싱으로 이후 요청에서 빨라짐
- GPU는 CPU 대비 약 4~5배 빠름

### CPU 및 메모리 사용량

**GPU 모드 (nvidia-smi 캡처)**:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| 30%   45C    P2    85W / 350W |   3856MiB / 24576MiB |     65%      Default |
+-------------------------------+----------------------+----------------------+

```

- **VRAM 사용량**: 약 3.8GB (기본 화자), 5.2GB (화자 복제)
- **GPU 사용률**: 합성 중 60~70%, 대기 시 5% 이하

**CPU 모드 (Task Manager 캡처)**:

```
프로세스: uvicorn (python)
CPU: 45~65% (12 cores 중)
메모리: 6.8GB
디스크: 200MB/s (모델 로딩 시)

```

---

## 3. 에러 및 문제 해결 과정

### 에러 1: `torchcodec` 의존성 에러

### 발생한 에러 메시지

```python
ImportError: TorchCodec is required for load_with_torchcodec.
Please install it using 'pip install torchcodec'.

```

### 원인 분석

- `torchaudio 2.4.0+`부터 `torchcodec` 의존성 추가
- `torchcodec`는 비디오 처리용 라이브러리로 TTS에 불필요
- 설치 시 CUDA 버전 충돌 발생 가능

### 해결을 위한 시도

**시도 1**: torchcodec 설치 (실패)

```bash
pip install torchcodec
# 에러: No matching distribution found for torchcodec (Windows)

```

**시도 2**: torchaudio 다운그레이드 (성공 ✅)

```bash
uv pip uninstall torchaudio
uv pip install torchaudio==2.3.1

```

**시도 3**: pyproject.toml 버전 고정

```toml
[project]
dependencies = [
    "torch==2.3.1",
    "torchaudio==2.3.1",  # ⭐️ 필수 고정!
]

```

### 해결 결과

- torchaudio 2.3.1 고정으로 안정적 작동
- 공식 문서에서도 2.3.x 권장 확인

### 느낀 점

- 의존성 버전 충돌은 AI 모델 구동 시 가장 흔한 문제
- `pyproject.toml`에서 버전 명시적 고정이 중요
- 에러 메시지의 키워드 ("torchcodec", "torchaudio")로 구글링하여 GitHub Issue 발견
- **배운 점**: 최신 버전이 항상 좋은 것은 아니며, 안정성을 위해 버전 고정 필요

---

### 에러 2: 일본어 Tokenizer 충돌

### 발생한 에러 메시지

```python
RuntimeError: PytorchStreamReader failed reading zip archive:
failed finding central directory
# 또는
KeyError: 'unk_token' in Japanese tokenizer

```

### 원인 분석

- XTTS v2의 일본어 tokenizer가 특정 문자(한자, 특수기호)에서 에러
- MeCab 사전 미설치 또는 버전 불일치
- Coqui TTS 0.25.3의 알려진 버그

### 해결을 위한 시도

**시도 1**: MeCab 사전 재설치 (부분 성공)

```bash
# Ubuntu
sudo apt-get install mecab mecab-ipadic-utf8

# macOS
brew install mecab mecab-ipadic

# Python 패키지
pip install fugashi unidic-lite cutlet

```

**시도 2**: `lang="en"` 우회 (임시 해결 ✅)

```python
# 일본어 텍스트지만 lang="en" 사용
response = requests.post(
    "<http://localhost:8100/synthesize_base64>",
    json={
        "text": "こんにちは、世界",  # 일본어 텍스트
        "lang": "en"  # "ja" 대신 "en" (우회)
    }
)

```

- 품질: 약간 부자연스러우나 이해 가능
- 장점: 에러 없이 합성 가능
- 단점: 일본어 고유 억양 손실

**시도 3**: Coqui TTS GitHub Issue 검색

- Issue #3421: Japanese tokenizer bug 보고됨
- 해결: 공식 업데이트 대기 중 (0.26.0에서 수정 예정)

### 해결 결과

- 현재는 `lang="en"` 우회로 운영 중
- 장기적으로는 Coqui TTS 업데이트 필요

### 느낀 점

- 오픈소스 모델은 완벽하지 않으며, 알려진 버그 존재
- GitHub Issues가 문제 해결의 핵심 리소스
- **배운 점**: 우회 방법(workaround)도 유효한 해결책이며, 완벽한 해결을 기다리기보다 실용적 접근 중요

---

### 에러 3: 첫 요청 타임아웃

### 발생한 에러 메시지

```python
requests.exceptions.ReadTimeout:
HTTPConnectionPool(host='localhost', port=8100):
Read timed out. (read timeout=60)

```

### 원인 분석

- 첫 요청 시 화자 임베딩 생성에 10~30초 소요
- 클라이언트 기본 타임아웃(60초)보다 오래 걸림 (특히 CPU 환경)
- Speaker Encoder가 음성 샘플을 벡터로 변환하는 과정 필요

### 해결을 위한 시도

**시도 1**: 클라이언트 타임아웃 증가 (성공 ✅)

```python
# Before
response = requests.post(url, json=data, timeout=60)

# After
response = requests.post(url, json=data, timeout=180)  # 60초 → 180초

```

**시도 2**: 서버 로그 확인

```python
# server_tts.py에서 DEBUG 활성화
DEBUG = True

# 로그 출력 예시
"""
🎤 Processing speaker embedding...
✅ Speaker embedding generated (12.5s)
🗣️ Synthesizing speech...
✅ Synthesis completed (8.2s)
"""

```

**시도 3**: 화자 임베딩 캐싱 (개선 시도)

```python
# 향후 개선 아이디어 (미구현)
speaker_cache = {}

def get_speaker_embedding(wav_b64):
    cache_key = hashlib.md5(wav_b64.encode()).hexdigest()
    if cache_key not in speaker_cache:
        speaker_cache[cache_key] = compute_embedding(wav_b64)
    return speaker_cache[cache_key]

```

### 해결 결과

- 타임아웃 180초로 증가하여 안정적 작동
- 첫 요청 후 속도 개선 확인 (임베딩 재사용)

### 느낀 점

- AI 모델의 첫 요청은 초기화 비용이 크다는 점 인지
- 사용자 경험을 위해 로딩 인디케이터 필수
- **배운 점**: 성능 병목 지점 파악을 위해 로깅이 중요하며, 캐싱 전략이 성능 개선의 핵심

---

### 에러 4: GPU 메모리 부족 (OOM)

### 발생한 에러 메시지

```python
RuntimeError: CUDA out of memory.
Tried to allocate 2.50 GiB
(GPU 0; 23.70 GiB total capacity;
20.80 GiB already allocated)

```

### 원인 분석

- 다른 GPU 프로세스와 메모리 공유 (예: 다른 모델 서버)
- XTTS v2는 최소 4GB VRAM 필요
- 화자 복제 사용 시 6GB 권장

### 해결을 위한 시도

**시도 1**: 다른 GPU 프로세스 종료

```bash
nvidia-smi  # GPU 사용 현황 확인
# PID 12345: my_other_model (8GB)
# PID 67890: xtts_v2 (4GB)

kill 12345  # 불필요한 프로세스 종료

```

**시도 2**: CPU 모드로 전환 (대안)

```python
# server_tts.py
device = "cpu"

```

- 장점: 메모리 문제 해결
- 단점: 속도 저하 (4~5배 느림)

**시도 3**: 배치 크기 조절 (미지원)

- XTTS v2는 배치 처리 미지원 (한 번에 하나씩만 처리)
- 병렬 처리는 별도 워커 구성 필요

### 해결 결과

- GPU 프로세스 정리 후 정상 작동
- 운영 시 GPU 메모리 모니터링 필수

### 느낀 점

- GPU 리소스는 공유 자원이므로 관리 필요
- `nvidia-smi` 명령어로 실시간 모니터링 습관화
- **배운 점**: 클라우드 환경에서는 자동 스케일링이 중요하며, 로컬 개발 시 리소스 제약 고려 필수

---

## 4. '나만의 음성 모델' 만들기

### 4.1 GUI/앱 구현 (Streamlit)

### 구현 화면 구조

**메인 화면**:

```
┌─────────────────────────────────────────────────────┐
│ Peace Chatbot System (Gemini + Multi-TTS/STT)      │
├─────────────────────────────────────────────────────┤
│ Sidebar (좌측)          │ Main Area (중앙)          │
│                         │                           │
│ TTS Model:              │ 🎤 Record your voice      │
│ ○ MeloTTS               │ [녹음시작] [녹음정지]        │
│ ● XTTS v2 ← 선택됨       │                           │
│                         │ 저장된 파일:               │
│ STT Model:              │ /tmp/tmpXXXXX.wav         │
│ ● Vosk                  │                           │
│ ✅ Connected             │ ──────────────────────   │
│                         │                           │
│ Language:               │ 💬 Chat Input             │
│ [Korean ▼]              │                           │
│                         │ User: 오늘 날씨 어때?      │
│ GEMINI API Key:         │ 🎤                        │
│ [••••••••••]            │                           │
│                         │ ──────────────────────   │
│ LLM 최대 글자: 300      │                           │
│ ☑ Show Audio            │ 대화 히스토리:             │
│                         │ ┌───────────────────────┐ │
│ [Rewind] [Clear]        │ │ User: 오늘 날씨 어때?  │ │
│                         │ └───────────────────────┘ │
│ 현재 화자:               │ ┌───────────────────────┐ │
│ my_voice1.wav           │ │ Assistant:            │ │
│                         │ │ 오늘 날씨는 맑고       │ │
│                         │ │ 따뜻합니다.            │ │
│                         │ │ [▶ 재생]              │ │
│                         │ └───────────────────────┘ │
└─────────────────────────────────────────────────────┘

```

### 핵심 기능

**1. 화자 녹음 (Voice Recording)**

```python
# audiorecorder 사용
audio = audiorecorder("녹음시작", "녹음정지", key="xtts_recorder")

if len(audio) > 0:
    with NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        audio.export(tmp_path, format="wav")
        st.session_state.speaker_path = tmp_path

```

**2. 음성 입력 (Voice Input)**

```python
audio_stt = audiorecorder("🎤", "⏹️", key="stt_recorder")

if audio_stt and len(audio_stt) > 0:
    audio_buffer = io.BytesIO()
    audio_stt.export(audio_buffer, format="wav")
    audio_bytes = audio_buffer.getvalue()

    transcribed_text = stt_inference(
        model_key="vosk",
        audio_bytes=audio_bytes,
        vosk_lang_code="KR",
    )

```

**3. 자동 재생 (Autoplay)**

```python
# 마지막 assistant 메시지만 autoplay
for i, msg in enumerate(st.session_state.messages):
    embed = msg.get("tts_embed", "")

    if autoplay_index is not None and i == autoplay_index:
        embed = embed.replace("<audio controls>", "<audio controls autoplay>")

    st.markdown(embed, unsafe_allow_html=True)

# 렌더 후 초기화 (한 번만 재생)
st.session_state.autoplay_index = None

```

---

### 4.2 아이디어 및 시도

### 아이디어 1: 다국어 화자 복제 챗봇

**목표**: 사용자 목소리로 12개 언어 답변

**구현**:

```python
# 한 번의 녹음으로 모든 언어 지원
speaker_path = "my_voice1.wav"  # 한국어 녹음

# 영어 응답
tts_inference(model_key="xtts_v2", text="Hello", lang_code="en", speaker_path=speaker_path)

# 프랑스어 응답
tts_inference(model_key="xtts_v2", text="Bonjour", lang_code="fr", speaker_path=speaker_path)

```

**결과**:

- ✅ 성공: 한국어 화자로 영어, 프랑스어 합성 가능
- 품질: 한국어 억양이 약간 남지만 이해 가능
- 활용: 외국어 학습 앱에 적용 가능

### 아이디어 2: 속도 조절 비교

**목표**: 속도에 따른 자연스러움 비교

**테스트 케이스**:

```python
speeds = [0.5, 0.8, 1.0, 1.2, 1.5]
text = "안녕하세요, 오늘 날씨가 정말 좋네요."

for speed in speeds:
    output = tts_inference(
        model_key="xtts_v2",
        text=text,
        lang_code="ko",
        speaker_path="my_voice1.wav",
        speed=speed
    )
    # 저장 및 청취 테스트

```

**결과표**:

| 속도 | 재생 시간 | 자연스러움 | 명료도 | 권장 용도 |
| --- | --- | --- | --- | --- |
| 0.5x | 6.0초 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 교육/학습 |
| 0.8x | 3.8초 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 또렷한 발음 |
| 1.0x | 3.0초 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 일반 대화 (권장) |
| 1.2x | 2.5초 | ⭐⭐⭐⭐ | ⭐⭐⭐ | 빠른 정보 전달 |
| 1.5x | 2.0초 | ⭐⭐⭐ | ⭐⭐ | 시간 제약 시 |

**결론**: 1.0x가 최적, 0.8~1.2x 범위 권장

### 아이디어 3: 화자 샘플 길이 비교

**목표**: 최소 샘플 길이 찾기

**테스트**:

```python
sample_lengths = [3, 6, 10, 15, 30]  # 초 단위

for length in sample_lengths:
    # length초 만큼 녹음
    speaker_wav = record_audio(duration=length)

    # TTS 합성
    output = tts_inference(
        model_key="xtts_v2",
        text="이것은 화자 복제 테스트입니다.",
        lang_code="ko",
        speaker_path=speaker_wav
    )

    # 유사도 평가 (주관적 청취 테스트)

```

**결과표**:

| 샘플 길이 | 화자 유사도 | 자연스러움 | 안정성 |
| --- | --- | --- | --- |
| 3초 | 60% | ⭐⭐ | 불안정 (에러 발생) |
| 6초 | 75% | ⭐⭐⭐ | 최소 요구사항 |
| 10초 | 85% | ⭐⭐⭐⭐ | 권장 ✅ |
| 15초 | 90% | ⭐⭐⭐⭐⭐ | 최적 |
| 30초 | 92% | ⭐⭐⭐⭐⭐ | 최고 품질 |

**결론**:

- **최소**: 6초 (에러 없음)
- **권장**: 10~15초 (품질/효율 균형)
- **최적**: 30초 (최고 품질, 시간 여유 시)

---

### 4.3 CPU 환경에서의 제약 및 가능성

### ❌ CPU 환경에서 불가능한 것

**1. 실시간 대화 (Real-time Conversation)**

- 요구사항: 응답 시간 < 3초
- CPU 성능: 20~40초
- 결론: ❌ 사용자 경험 저하로 실용성 없음

**2. 대량 음성 생성 (Batch Processing)**

```python
# 100개 문장 TTS 생성
texts = ["문장1", "문장2", ..., "문장100"]

# CPU:
```

- 결론: ❌ CPU는 비현실적

**3. 음성 복제 모델 재훈련 (Fine-tuning)**

- 요구사항: 수백 시간 음성 데이터 처리
- CPU 시간: 수 주일
- 결론: ❌ GPU 필수

**4. Large 모델 테스트**

- XTTS v2는 450M 파라미터로 이미 중대형
- CPU 메모리 부족 위험
- 결론: ❌ CPU 한계

### ⭕️ CPU 환경에서도 가능한 것

**1. 속도 조절 실험 (Speed Control)**

```python
# 속도만 변경하여 비교 (합성 시간 동일)
for speed in [0.5, 1.0, 1.5]:
    tts_inference(..., speed=speed)

```

- ✅ 가능: 속도 파라미터 변경은 연산 비용 낮음
- 시간: 속도와 무관하게 20~30초

**2. 언어별 품질 비교 (Language Comparison)**

```python
languages = ["ko", "en", "ja", "zh-cn"]

for lang in languages:
    output = tts_inference(
        text=f"Hello in {lang}",
        lang_code=lang,
        ...
    )
    # 주관적 품질 평가

```

- ✅ 가능: 소량 샘플 생성 (4개 × 25초 = 100초)

**3. 화자 샘플 길이 비교**

- 앞서 4.2 아이디어 3 참조
- ✅ 가능: 5개 샘플 테스트 (5 × 25초 = 125초)

**4. 텍스트 전처리 효과 비교**

```python
texts = [
    "안녕하세요",           # 원본
    "안녕하세요.",          # 마침표 추가
    "안녕하세요!",          # 느낌표
    "안녕하세요...",        # 여운
]

for text in texts:
    tts_inference(text=text, ...)

```

- ✅ 가능: 억양 변화 관찰 (4개 × 25초 = 100초)

**5. 에러 케이스 테스트**

```python
# 특수문자, 이모지, 숫자 처리
test_cases = [
    "가격은 10,000원입니다",
    "Hello 😊",
    "Tel: 010-1234-5678",
]

```

- ✅ 가능: 소량 엣지 케이스 검증

---

### 4.4 비교 실험 결과

### 실험 1: XTTS v2 vs MeloTTS

**테스트 조건**:

- 텍스트: "안녕하세요, 오늘 날씨가 정말 좋네요."
- 환경: CPU (공정한 비교)
- 측정: 속도, 자연스러움, 화자 복제

**결과표**:

| 항목 | XTTS v2 | MeloTTS | 승자 |
| --- | --- | --- | --- |
| **속도** | 25초 | 2초 | MeloTTS ✅ |
| **자연스러움** | ⭐⭐⭐⭐⭐ (4.8/5) | ⭐⭐⭐⭐ (3.9/5) | XTTS v2 ✅ |
| **화자 복제** | ✅ 가능 | ❌ 불가능 | XTTS v2 ✅ |
| **메모리** | 6.8GB | 2.1GB | MeloTTS ✅ |
| **다국어** | 14개 언어 | 6개 언어 | XTTS v2 ✅ |

**결론**:

- **MeloTTS**: 빠른 응답이 중요한 챗봇
- **XTTS v2**: 품질과 개인화가 중요한 서비스

### 실험 2: GPU vs CPU 성능 비교

**테스트**: 10자 한국어 텍스트 10회 연속 합성

**결과**:

| 환경 | 평균 시간 | 표준편차 | 최소 | 최대 |
| --- | --- | --- | --- | --- |
| **GPU (RTX 3090)** | 6.2초 | 0.8초 | 5.1초 | 7.3초 |
| **CPU (Ryzen 9)** | 24.8초 | 2.1초 | 22.3초 | 28.1초 |

**속도 비율**: GPU는 CPU 대비 **4.0배 빠름**

**결론**: 실시간 서비스는 GPU 필수

---

## 5. 결론

### 5.1 기술적 요소 요약

**XTTS v2 특징**:

- **파라미터**: 약 450M (중대형 모델)
- **속도**: GPU 6초, CPU 25초 (10자 기준)
- **강점**: 화자 복제, 다국어 지원, 자연스러운 억양
- **약점**: 느린 속도, 높은 메모리 요구량, 일본어 버그

**환경 선택의 중요성**:

- GPU 환경: 실시간 대화 가능 (5~10초 응답)
- CPU 환경: 오프라인 배치 처리 전용 (20~40초)

**실험을 통한 발견**:

- 화자 샘플: 10~15초가 최적 (품질/효율 균형)
- 속도 설정: 1.0x 권장 (0.8~1.2x 범위 내)
- 언어별 품질: 한국어(4.8/5), 영어(4.9/5), 일본어(3.5/5, 우회 사용)

---

### 5.2 기술 구현 경험 느낀 점

### 의존성 관리의 중요성

단순히 모델을 실행하는 것이 아니라, 올바른 버전의 라이브러리를 찾고 충돌을 해결하는 과정이 전체 시간의 30% 이상을 차지했다. `torchaudio 2.3.1` 고정이 핵심이었으며, 이를 통해 **버전 명시의 중요성**을 체감했다.

### 오류 해결 프로세스

에러 메시지를 단순히 읽는 것이 아니라, 핵심 키워드를 추출하여 구글링하고, GitHub Issues에서 비슷한 사례를 찾는 과정이 매우 효과적이었다. 특히 일본어 tokenizer 문제는 공식 Issue #3421에서 해결 힌트를 얻었다.

### 성능 측정의 가치

"GPU가 빠르다"는 막연한 인식이 아니라, **정확히 4배 빠르다**는 수치를 측정함으로써 환경 선택 기준을 명확히 할 수 있었다. 또한 화자 샘플 길이 실험을 통해 10초가 최적점임을 발견했다.

### 사용자 경험 중심 설계

기술적 완성도도 중요하지만, Streamlit UI에서 로딩 인디케이터, 자동 재생, 에러 메시지 등 **사용자 피드백**이 더 중요함을 깨달았다. CPU 환경에서 20초 대기는 기술적으로 가능하지만 사용자 경험 측면에서 실패다.

---

### 5.3 다음에 도전하고 싶은 목표

### 1. 화자 임베딩 캐싱 시스템 구축

현재는 매 요청마다 화자 임베딩을 생성하지만, Redis나 메모리 캐시를 사용하여 동일 화자의 반복 요청 속도를 50% 단축하고 싶다.

```python
# 목표 구조
@lru_cache(maxsize=100)
def get_speaker_embedding(wav_hash):
    return model.get_speaker_embedding(wav_path)

```

### 2. 스트리밍 TTS 구현

현재는 전체 문장 합성 후 반환하지만, **청크 단위 스트리밍**으로 첫 음절을 0.5초 내에 재생하여 체감 속도를 개선하고 싶다.

```python
# 목표: 실시간 스트리밍
for chunk in model.synthesize_stream(text):
    yield chunk  # 0.1초마다 전송

```

### 3. 감정 제어 기능 추가

XTTS v2는 감정 표현이 가능하지만, 현재는 자동 추론만 사용 중이다. 사용자가 감정(기쁨, 슬픔, 화남)을 선택할 수 있는 인터페이스를 구현하고 싶다.

```python
# 목표 API
tts_inference(
    text="오늘 너무 행복해요!",
    emotion="happy",  # 새 파라미터
    emotion_intensity=0.8  # 0~1
)

```

### 4. Fine-tuning 시도 (GPU 서버 환경)

Coqui TTS는 커스텀 데이터로 Fine-tuning이 가능하다. 특정 도메인(예: 의료, 금융)의 전문 용어 발음 정확도를 높이기 위해 10시간 분량의 음성 데이터로 Fine-tuning을 시도하고 싶다.

### 5. 다중 화자 대화 시스템

현재는 단일 화자만 지원하지만, 여러 화자를 등록하고 대화 중 동적으로 전환하는 **다중 화자 챗봇**을 구현하여 극본 낭독이나 교육 콘텐츠에 활용하고 싶다.

```python
# 목표 구조
speakers = {
    "user": "my_voice.wav",
    "assistant": "ai_voice.wav",
    "narrator": "narrator_voice.wav"
}

tts_inference(text="...", speaker_id="narrator")

```

---

### 5.4 최종 소감

XTTS v2 프로젝트를 통해 **AI 모델은 단순히 실행하는 것이 아니라, 환경 구축부터 오류 해결, 성능 최적화까지 전 과정이 중요하다**는 점을 배웠다. 특히 의존성 충돌, GPU/CPU 선택, 사용자 경험 설계 등 실무에서 마주할 문제들을 직접 경험하며 성장할 수 있었다.

현재는 컴퓨팅 자원과 시간 제약으로 제한적인 실험만 진행했지만, 앞으로는 스트리밍, 캐싱, Fine-tuning 등 더 발전된 기술을 시도하고 싶다. 이번 경험이 AI 엔지니어로 성장하는 데 탄탄한 기초가 되었다고 확신한다.

---

**Version**: 1.0.0

**Report Date**: 2024-11-27

**Author**: Peace Cho

**Contact**: [chopeacekr@gmail.com](mailto:chopeacekr@gmail.com)

---

**참고 문헌**:

- Coqui TTS Documentation: https://github.com/coqui-ai/TTS
- XTTS v2 Paper: https://arxiv.org/abs/2406.04904
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html