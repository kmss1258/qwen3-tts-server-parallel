# Qwen3-TTS Voice Design vs Voice Clone 입력 벡터 분석

> **분석 날짜**: 2026-02-02  
> **분석 대상**: Qwen3-TTS-12Hz 모델 아키텍처  
> **핵심 질문**: Voice Design 입력 벡터와 Voice Clone 입력 벡터를 동일하게 만들어 일관된 추론 결과를 얻을 수 있는가?

---

## 1. 요약 (TL;DR)

**결론: 호환 불가능**

Voice Design과 Voice Clone은 **완전히 다른 메커니즘**으로 화자(speaker)를 표현합니다:

| 특성 | Voice Design | Voice Clone (Base) |
|------|--------------|-------------------|
| **화자 표현** | 자연어 텍스트 설명 (instruct) | Speaker Embedding (x-vector, 1024D) |
| **입력 구조** | `[instruct_ids] + [text_ids] + [codec_tokens]` | `[speaker_embed] + [text_ids] + [codec_tokens]` |
| **결정론적** | ❌ (매번 다른 화자 생성) | ✅ (동일 임베딩 = 동일 화자) |
| **모델 타입** | `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | `Qwen3-TTS-12Hz-1.7B-Base` |

**Voice Design은 설계상 매 생성마다 다른 목소리를 만들도록 의도**되었습니다. 일관된 목소리가 필요하면 Voice Clone을 사용해야 합니다.

---

## 2. 아키텍처 심층 분석

### 2.1 공통 아키텍처

두 모드 모두 **Qwen3TTSForConditionalGeneration** 기반이며 다음 구조를 공유:

```
                    ┌──────────────────────────────────────┐
                    │  Talker (Qwen3TTSTalkerForConditionalGeneration)  │
                    │                                      │
  Input Embeds ──► │  ┌─────────────────────────────────┐  │
                    │  │  text_projection (MLP)          │  │
                    │  │  text_embedding (Vocab→D)       │  │
                    │  │  codec_embedding (Vocab→D)      │  │
                    │  │  speaker_embed (optional)       │  │
                    │  └─────────────────────────────────┘  │
                    │              ↓                        │
                    │  ┌─────────────────────────────────┐  │
                    │  │  20 x Qwen3TTSTalkerDecoderLayer│  │
                    │  │  (Multi-head Attention + MLP)   │  │
                    │  └─────────────────────────────────┘  │
                    │              ↓                        │
                    │  codec_head → 음성 코덱 토큰 예측    │
                    └──────────────────────────────────────┘
                                   ↓
                    ┌──────────────────────────────────────┐
                    │  Speech Tokenizer (Code2Wav)         │
                    │  → Waveform 생성                     │
                    └──────────────────────────────────────┘
```

### 2.2 Voice Design 입력 벡터 구성

**모델 타입**: `tts_model_type = "voice_design"`

```python
# modeling_qwen3_tts.py: generate() 함수 분석

def generate(self, ..., instruct_ids=None, ...):
    # 1. Instruct 텍스트 임베딩 (자연어 화자 설명)
    if instruct_ids is not None:
        for index, instruct_id in enumerate(instruct_ids):
            if instruct_id is not None:
                talker_input_embeds[index].append(
                    self.talker.text_projection(
                        self.talker.get_text_embeddings()(instruct_id)
                    )
                )
    
    # 2. 코덱 프리필 (언어 태그만, speaker_embed 없음!)
    codec_prefill_list = [[
        self.config.talker_config.codec_think_id,      # 4202
        self.config.talker_config.codec_think_bos_id,  # 4204
        language_id,                                    # 언어 ID
        self.config.talker_config.codec_think_eos_id,  # 4205
    ]]
    
    # 3. 최종 입력 구조
    # [instruct_embed] + [codec_tags] + [text_embed] → 생성
```

**Voice Design 입력 시퀀스**:
```
┌────────────────┬─────────────────┬──────────────────┬────────────────┐
│  Instruct      │  Language Tags  │  TTS BOS/PAD     │  Text Content  │
│  (자연어 설명)  │  (Think tokens) │  Tokens          │  (읽을 텍스트) │
└────────────────┴─────────────────┴──────────────────┴────────────────┘
        ↓                ↓                   ↓                ↓
   text_projection  codec_embedding   codec_embedding   text_projection
        ↓                ↓                   ↓                ↓
        └────────────────┴───────────────────┴────────────────┘
                                ↓
                    inputs_embeds (hidden_size=1024)
```

### 2.3 Voice Clone 입력 벡터 구성

**모델 타입**: `tts_model_type = "base"`

#### 2.3.1 Speaker Embedding (x-vector) 추출

```python
# modeling_qwen3_tts.py

def extract_speaker_embedding(self, audio, sr):
    """ECAPA-TDNN 기반 화자 임베딩 추출"""
    assert sr == 24000  # 24kHz 필수
    
    # Mel spectrogram 변환
    mels = mel_spectrogram(
        torch.from_numpy(audio).unsqueeze(0),
        n_fft=1024, num_mels=128, sampling_rate=24000,
        hop_size=256, win_size=1024, fmin=0, fmax=12000
    ).transpose(1, 2)
    
    # ECAPA-TDNN Speaker Encoder로 임베딩 추출
    speaker_embedding = self.speaker_encoder(mels.to(self.device).to(self.dtype))[0]
    return speaker_embedding  # Shape: (1024,)
```

**Speaker Encoder 아키텍처** (ECAPA-TDNN):
```
┌────────────────────────────────────────────────────────────────┐
│                  Qwen3TTSSpeakerEncoder                        │
│  Input: Mel Spectrogram (128 mels)                             │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  TimeDelayNetBlock (128 → 512)                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  3 x SqueezeExcitationRes2NetBlock                      │   │
│  │  (512 → 512, kernel=3, dilation=2,3,4)                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Multi-layer Feature Aggregation (MFA)                  │   │
│  │  Concat all layer outputs → TDNN (1536 → 1536)          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  AttentiveStatisticsPooling (1536 → 3072)               │   │
│  │  Mean + Std concatenation                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Final Conv1d (3072 → 1024)                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
│  Output: speaker_embedding (1024,)                             │
└────────────────────────────────────────────────────────────────┘
```

#### 2.3.2 Voice Clone 입력 시퀀스

**X-Vector Only 모드** (`x_vector_only_mode=True`):
```
┌────────────────┬──────────────────┬────────────────┬────────────────┐
│  Language Tags │  Speaker Embed   │  TTS BOS/PAD   │  Text Content  │
│  (Think tokens)│  (1024D vector)  │  Tokens        │  (읽을 텍스트) │
└────────────────┴──────────────────┴────────────────┴────────────────┘
        ↓                ↓                   ↓                ↓
   codec_embedding  (직접 주입)      codec_embedding   text_projection
```

**ICL 모드** (`x_vector_only_mode=False`):
```
┌────────────────┬──────────────────┬──────────────────┬────────────────┐
│  Language Tags │  Speaker Embed   │  Reference Audio │  Text Content  │
│  (Think tokens)│  (1024D vector)  │  (ref_code+text) │  (읽을 텍스트) │
└────────────────┴──────────────────┴──────────────────┴────────────────┘
        ↓                ↓                   ↓                ↓
   codec_embedding  (직접 주입)     generate_icl_prompt  text_projection
```

---

## 3. 핵심 차이점 상세 분석

### 3.1 화자 정보 인코딩 방식

| 항목 | Voice Design | Voice Clone |
|------|--------------|-------------|
| **화자 정보 소스** | 자연어 텍스트 | 오디오 샘플 |
| **인코딩 방식** | Text Tokenizer → text_projection | Mel Spec → ECAPA-TDNN |
| **임베딩 차원** | text_hidden_size=2048 → 1024 | 직접 1024D |
| **주입 위치** | 시퀀스 앞에 concat | codec_embedding 위치에 직접 주입 |

### 3.2 코드 레벨 비교

**Voice Design (modeling_qwen3_tts.py:2086-2106)**:
```python
# speaker_embed은 항상 None (Voice Design 모델)
if voice_clone_spk_embeds is None:
    if speaker == "" or speaker == None:
        speaker_embed = None  # Voice Design은 여기!
    else:
        # CustomVoice 모델의 미리 정의된 화자
        spk_id = self.config.talker_config.spk_id[speaker.lower()]
        speaker_embed = self.talker.get_input_embeddings()(...)
```

**Voice Clone (modeling_qwen3_tts.py:2102-2106)**:
```python
else:
    # voice_clone_spk_embeds가 있는 경우 (Voice Clone 모델)
    if voice_clone_prompt["x_vector_only_mode"][index] or voice_clone_prompt["icl_mode"][index]:
        speaker_embed = voice_clone_spk_embeds[index]  # 추출된 임베딩 사용
```

### 3.3 결정론성 비교

**Voice Design**:
- 동일한 instruct 텍스트로도 **매번 다른 화자** 생성
- `do_sample=True` (기본값)로 인한 확률적 샘플링
- 모델이 instruct를 "해석"하여 화자 특성 생성
- **Think Tokens**로 복잡한 설명 처리

**Voice Clone**:
- 동일한 speaker_embedding → **동일한 화자 특성**
- 오디오에서 추출한 1024D 벡터가 화자 정체성 결정
- 수치적으로 동일한 임베딩 = 동일한 화자

---

## 4. Voice Design의 일관성 확보 방안

### 4.1 방법 1: Voice Design → Voice Clone 변환 (권장)

```python
# 1. Voice Design으로 원하는 목소리 생성
wavs, sr = model.generate_voice_design(
    text="안녕하세요, 저는 AI입니다.",
    instruct="밝고 활기찬 20대 여성 목소리로",
    language="Korean"
)

# 2. 생성된 음성에서 speaker embedding 추출
speaker_embed = model.extract_speaker_embedding(
    audio=wavs[0], 
    sr=24000  # resample if needed
)

# 3. 이후 Voice Clone Base 모델에서 동일 임베딩 사용
voice_clone_prompt = {
    "ref_spk_embedding": [speaker_embed],
    "x_vector_only_mode": [True],
    "icl_mode": [False],
    "ref_code": [None]
}

wavs2, sr = base_model.generate(
    input_ids=...,
    voice_clone_prompt=voice_clone_prompt,
    ...
)
```

### 4.2 방법 2: Sampling 고정

```python
# 결정론적 생성 시도 (완전히 동일하진 않음)
wavs, sr = model.generate_voice_design(
    text="안녕하세요",
    instruct="밝은 여성 목소리",
    do_sample=False,  # greedy decoding
    temperature=0.0
)
```

**주의**: 이 방법은 화자 일관성을 완전히 보장하지 않습니다. Think Tokens 처리 과정에서 여전히 변동이 있을 수 있습니다.

### 4.3 방법 3: Fine-tuning으로 커스텀 화자 임베딩 생성

참고: `finetuning/sft_12hz.py`

```python
# Fine-tuning 과정에서 target_speaker_embedding 저장
speaker_embedding = model.speaker_encoder(ref_mels).detach()

# 학습 후 codec_embedding.weight에 저장
state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0]
```

---

## 5. 입력 벡터 상세 스펙

### 5.1 공통 토큰 ID

```python
# Qwen3TTSConfig (configuration_qwen3_tts.py)
tts_pad_token_id = 151671
tts_bos_token_id = 151672
tts_eos_token_id = 151673

# Qwen3TTSTalkerConfig
codec_eos_token_id = 4198
codec_think_id = 4202
codec_nothink_id = 4203
codec_think_bos_id = 4204
codec_think_eos_id = 4205
codec_pad_id = 4196
codec_bos_id = 4197
```

### 5.2 Hidden Dimensions

```python
# Talker 모델 차원
hidden_size = 1024          # Talker hidden dim
text_hidden_size = 2048     # Text embedding dim (projection needed)
vocab_size = 3072           # Codec vocabulary
num_code_groups = 32        # Multi-codebook groups
num_hidden_layers = 20      # Transformer layers

# Speaker Encoder 출력
enc_dim = 1024              # Speaker embedding dimension
```

### 5.3 VoiceClonePromptItem 구조

```python
@dataclass
class VoiceClonePromptItem:
    ref_code: Optional[torch.Tensor]      # (T, Q) - ICL mode only
    ref_spk_embedding: torch.Tensor       # (1024,) - Always required
    x_vector_only_mode: bool              # True: x-vector only, False: ICL
    icl_mode: bool                        # Inverse of x_vector_only_mode
    ref_text: Optional[str] = None        # ICL mode: required
```

---

## 6. 결론 및 권장사항

### 6.1 질문에 대한 답변

**Q1: Voice Design 입력 벡터와 Voice Clone 입력 벡터를 똑같이 만들 수 있는가?**

**A: 불가능합니다.**
- Voice Design: 자연어 → text_projection → hidden_size
- Voice Clone: Audio → ECAPA-TDNN → speaker_embedding (1024D)
- 두 경로의 임베딩 공간이 완전히 다름

**Q2: Voice Design과 Voice Clone의 네트워크 입력 벡터 구성은?**

**A: 위 Section 2.2, 2.3 참조**
- Voice Design: `[instruct_embed] + [lang_tags] + [text_embed]`
- Voice Clone: `[lang_tags] + [speaker_embed] + [optional_icl] + [text_embed]`

### 6.2 권장 사용 패턴

1. **프로토타이핑**: Voice Design으로 원하는 목소리 탐색
2. **프로덕션**: 생성된 음성에서 speaker_embedding 추출
3. **일관된 생성**: 추출된 임베딩으로 Voice Clone 사용

```python
# 추천 워크플로우
def create_consistent_voice(instruct: str, reference_text: str):
    # 1. Voice Design으로 샘플 생성
    design_wav, sr = voice_design_model.generate_voice_design(
        text=reference_text,
        instruct=instruct
    )
    
    # 2. Speaker embedding 추출 및 저장
    speaker_emb = base_model.extract_speaker_embedding(
        design_wav, sr=24000
    )
    torch.save(speaker_emb, "my_speaker.pt")
    
    # 3. 이후 모든 생성에 저장된 임베딩 사용
    return speaker_emb

# 사용
speaker = create_consistent_voice(
    "따뜻하고 차분한 30대 남성",
    "안녕하세요"
)

# 일관된 추론
for text in texts:
    wav, sr = base_model.generate_voice_clone(
        text=text,
        voice_clone_prompt={
            "ref_spk_embedding": [speaker],
            "x_vector_only_mode": [True],
            ...
        }
    )
```

---

## 7. 참고 파일

- `qwen_tts/core/models/modeling_qwen3_tts.py` - 핵심 모델 구현
- `qwen_tts/core/models/configuration_qwen3_tts.py` - 설정 클래스
- `qwen_tts/inference/qwen3_tts_model.py` - 고수준 API
- `finetuning/sft_12hz.py` - Fine-tuning 예제
- `assets/Qwen3_TTS.pdf` - 공식 논문

---

*이 문서는 Qwen3-TTS 소스 코드 분석을 기반으로 작성되었습니다.*
