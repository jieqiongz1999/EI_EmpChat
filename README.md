# EI-EmpChat

> **EMNLP 2026** — Code and benchmark for the paper:  
> *"When Users Suppress and Models Flatter: Correcting Dual Emotional Distortion in Emotional Support Conversations"*

---

## Overview

Emotional support conversation (ESC) systems assume that expressed emotions faithfully reflect inner states — an assumption violated by **expressive suppression**, where distress is masked by hedged language and topic deflection. Compounding this, LLMs exhibit an intrinsic **optimism bias** that inflates response valence most severely when users need accurate empathy most.

EI-EmpChat addresses both problems through a three-stage pipeline:

```
Dialog Input
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ EIP  (Emotion Incongruence Perceiver)               │
│   → Detects expressive suppression                   │
│   → Outputs EI_Score ∈ [0,1] (incongruence score)   │
└──────────────────────┬──────────────────────────────┘
                       │ EI_Score
                       ▼
┌─────────────────────────────────────────────────────┐
│ ICETR  (Incongruence-aware Causal Emotion            │
│         Trajectory Reconstructor)                    │
│   → Extracts objective events from dialog history    │
│   → Reconstructs true emotion via temporal weighting │
│   → Fuses: CER = (1−EI_Score)·e_surf + EI_Score·e_recon │
└──────────────────────┬──────────────────────────────┘
                       │ CER, EI_Score, e_surf
                       ▼
┌─────────────────────────────────────────────────────┐
│ CBD-Gen  (Contrastive Bias-Decoupled Generation)     │
│   → LLaMA-3.1-8B + LoRA, CER injected via system prompt│
│   → Alignment penalty: penalizes optimistic training │
│   → Contrastive decoding: log p_final =              │
│         log p_expert − α·σ(−CER)·log p_surface       │
│   → EI_Score gate: contrast only when suppression    │
│         detected                                     │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
             Empathetic Support Response
```

We also contribute **ESConv-Incongruence**, the first benchmark annotating suppression phenomena in supportive dialogue (864 dialogs, IL ∈ {0,1,2}, LTE ∈ [-1,1]).

---

## Repository Structure

```
EI-EmpChat/
├── data/
│   └── ESConv-Incongruence/          # Benchmark (see data/README)
│       ├── README.md
│       └── sample/                   # 6 illustrative examples
│
├── eip/                              # Module 1: Emotion Incongruence Perceiver
│   ├── config.yaml
│   ├── train_bert.py
│   ├── evaluate_bert.py
│   └── eip/
│       ├── encoder.py                # EIPClassifier, EIPBertModel
│       └── bert_dataset.py           # BertEIPDataset, format_dialog_input
│
├── icetr/                            # Module 2: Emotion Trajectory Reconstructor
│   ├── config.yaml
│   ├── run_icetr.sh
│   ├── step1_extract_events.py
│   ├── step2_train_attribution.py
│   ├── step2_evaluate_attribution.py
│   ├── infer_icetr.py
│   └── icetr/
│       ├── event_extractor.py        # EventExtractor (stored / Qwen-Max API)
│       ├── attribution_model.py      # AttributionModel, AttributionWrapper
│       ├── attribution_dataset.py    # AttributionDataset
│       └── cer_fusion.py             # temporal_weights, reconstruct_emotion, cer_fusion
│
├── cbdgen/                           # Module 3: Contrastive Bias-Decoupled Generation
│   ├── config.yaml
│   ├── run_cbdgen.sh
│   ├── prepare_train_data.py         # Pre-compute ICETR + v_ref
│   ├── train_cbdgen.py               # LoRA fine-tuning with alignment penalty
│   ├── evaluate_cbdgen.py            # Full evaluation (contrastive decoding)
│   ├── evaluate_baseline.py          # Zero-shot LLM baseline
│   ├── evaluate_uniform_neutral.py   # Ablation: uniform anti-optimism prompt
│   ├── compare_results.py            # Aggregate results table
│   ├── compare_il_ablation.py        # IL-stratified Emot-MAE ablation
│   └── cbdgen/
│       ├── model.py                  # CBDGenModel, build_lora_model, prompts
│       ├── dataset.py                # CBDGenDataset, InferenceDataset
│       ├── contrastive.py            # contrastive_generate, standard_generate
│       └── metrics.py                # BLEU, Distinct, ROUGE-L, METEOR, BERTScore, Emot-MAE
│
├── baselines/
│   └── cso/                          # CSO (MCTS-based planning baseline)
│       ├── MCTS.py
│       ├── run.py
│       ├── util.py
│       ├── build_data.py
│       ├── prepare_esconv_subsets.py
│       └── evaluate_esconv_subsets.py
│
├── requirements.txt
└── README.md
```

---

## Prerequisites

### 1. Install dependencies

```bash
pip install -r requirements.txt
# NLTK data for METEOR
python -c "import nltk; nltk.download('punkt')"
```

### 2. Download base models

| Model | Purpose | HuggingFace ID |
|-------|---------|----------------|
| RoBERTa-large | EIP backbone | `roberta-large` |
| RoBERTa-base | ICETR attribution | `roberta-base` |
| Meta-Llama-3.1-8B-Instruct | CBD-Gen generator | `meta-llama/Meta-Llama-3.1-8B-Instruct` |

Update the `model_path` / `llm` fields in each module's `config.yaml` to point to your local copies.

### 3. Obtain the dataset

The full **ESConv-Incongruence** dataset (`train.json`, `dev.json`, `test.json`) is available upon request. Place the files under `data/ESConv-Incongruence/`. See `data/ESConv-Incongruence/README.md` for the schema.

---

## Training & Evaluation

### Stage 1 — EIP (Emotion Incongruence Perceiver)

```bash
cd eip

# Train
python train_bert.py --config config.yaml

# Evaluate
python evaluate_bert.py --config config.yaml --split test
```

Checkpoint saved to `eip/checkpoints/eip_bert_best.pt`.

---

### Stage 2 — ICETR (Incongruence-aware Causal Emotion Trajectory Reconstructor)

```bash
cd icetr

# Full pipeline (uses stored Qwen-Max annotations, no API call needed)
bash run_icetr.sh

# Or step by step:
python step1_extract_events.py --config config.yaml
python step2_train_attribution.py --config config.yaml
python step2_evaluate_attribution.py --config config.yaml --split test
python infer_icetr.py --config config.yaml --split test

# Interactive mode
python infer_icetr.py --config config.yaml --interactive
```

> **Note**: `step1_extract_events.py` by default reads `cot_reasoning.step1_events` from the annotated data (fast, free). Pass `--use_api` to call Qwen-Max for new dialogs (requires `DASHSCOPE_API_KEY`).

Checkpoint saved to `icetr/checkpoints/attribution_best.pt`.

---

### Stage 3 — CBD-Gen (Contrastive Bias-Decoupled Generation)

```bash
cd cbdgen

# Full pipeline
bash run_cbdgen.sh all

# Or step by step:
python prepare_train_data.py --config config.yaml        # pre-compute CER/EIS/v_ref
python train_cbdgen.py --config config.yaml              # LoRA fine-tuning
python evaluate_cbdgen.py --config config.yaml           # contrastive evaluation
python evaluate_cbdgen.py --config config.yaml --no_contrast  # ablation: no contrast
python evaluate_uniform_neutral.py --config config.yaml  # ablation: uniform prompt
python evaluate_baseline.py --config config.yaml \
    --model_path /path/to/Meta-Llama-3-8B-Instruct \
    --model_name llama3_8b                               # zero-shot baseline
```

#### Ablation analysis

```bash
python compare_il_ablation.py --config config.yaml       # IL-stratified Emot-MAE table
python compare_results.py --config config.yaml           # full results summary
```

---

### CSO Baseline

```bash
cd baselines/cso
# Requires OpenAI API key for GPT-4o-mini
export OPENAI_API_KEY="your_key"
python prepare_esconv_subsets.py
python run.py
python evaluate_esconv_subsets.py
```

---

## Key Components

### EI_Score (Emotion Incongruence Score)

Output of EIP module. Continuous score ∈ [0, 1], computed as:

```
EI_Score = σ(MLP([S_a; S_t; S_p; h_ctx]))
```

where S_a, S_t, S_p are the three suppression signals (affect–situation gap, turn-level emotional attenuation, pragmatic avoidance) and h_ctx is the dialogue context representation from RoBERTa-large.

### CER (Corrected Emotion Representation)

Fused latent emotion estimate from ICETR:

```
CER = (1 − EI_Score) × e_surf + EI_Score × e_recon
```

where `e_recon` is the temporally-weighted average of per-event valences:

```
e_recon = Σ_i  w_i × v_i,    w_i ∝ exp(−γ(T−i))
```

### Alignment Penalty (Training)

```
L = L_LM + λ · EI_Score · max(0, v_ref − CER)
```

Adds an asymmetric penalty when the reference response valence exceeds CER (i.e., the supporter was more positive than the corrected emotion warrants). Scaled by EI_Score to focus regularization on high-distortion samples.

### Contrastive Decoding (Inference)

```
log p_final = log p_expert − α(CER) · log p_surface
α(CER) = α₀ · σ(−CER)
```

Gated by EI_Score: contrastive decoding activates only when EI_Score exceeds a threshold and the CER--surface gap exceeds a minimum, avoiding overhead on congruent samples.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| ROUGE-L | LCS-based F1 (lexical overlap) |
| BLEU-3/4 | N-gram precision vs. reference |
| Distinct-1 | Lexical diversity |
| METEOR | Stem+exact match F1 (semantic similarity) |
| Extrema | Embedding-based sentence similarity |
| CIDEr | Consensus with reference responses |
| **Emot-MAE** | `\|v(generated) − CER\|` — primary emotion alignment metric |

All metrics reported stratified by IL level (IL=0/1/2) to verify that EI-EmpChat corrects suppressed cases without over-correcting congruent ones.

---

## Citation

```bibtex
@inproceedings{eiemp2026,
  title     = {When Users Suppress and Models Flatter: Correcting Dual
               Emotional Distortion in Emotional Support Conversations},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods
               in Natural Language Processing (EMNLP)},
  year      = {2026}
}
```

---

## License

Code: MIT License.  
ESConv-Incongruence dataset: CC BY-NC 4.0 (research use only).
