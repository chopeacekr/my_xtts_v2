# 🗣️ XTTS v2 — 요약

## 1. 데이터 특성

- 12개 언어 기반 멀티스피커 코퍼스
- 스튜디오 녹음 + 일반 음성 포함
- 3~6초 보이스 클로닝 가능
- 언어별 G2P/phoneme 변환 사용

## 2. 학습 방식 / 모델 구조

- Cross-lingual TTS (음색 유지 + 다른 언어 발화)
- VITS 기반 Non-autoregressive 구조
- Speaker Encoder(d-vector), Language Embedding
- Transformer Text Encoder + Flow Acoustic Model
- HiFi-GAN Vocoder 사용

## 3. 사용 라이브러리

- PyTorch(torch, torchaudio)
- transformers
- librosa, soundfile
- g2p_en, g2pk, pypinyin, MeCab 등 G2P
- Coqui-TTS 프레임워크

## 4. 모델 크기

- 전체 모델: 약 1.3~1.5GB
- Acoustic/Text: ~800MB
- Vocoder: 200~300MB
- Speaker Encoder: ~100MB

## 5. 언어 지원

- 총 12개 언어(EN, ZH, KO, JP, ES, FR, DE, IT, PT, TR, PL, RU)
- 모든 언어에서 보이스 클로닝과 cross-lingual TTS 가능