# GenAI approach NER Pipeline (Local LoRA / GPT-4o-mini)

This project extracts **official/person entities + their actions/events** from *Ming Shilu* PDF pages and writes them as structured **JSONL** records (one JSON object per line).

You get **two interchangeable inference backends**:

* **Approach A — Local model (MLX LoRA / Ollama fallback)**: runs on your machine, optimized for Apple Silicon when using MLX.
* **Approach B — GPT-4o-mini via OpenAI API**: higher baseline accuracy, less local setup, costs API usage.

---

## 1) Project layout

Core scripts:

* `runNER.py` — Local inference pipeline (MLX LoRA by default; can switch to Ollama).
* `runNER_GPT.py` — OpenAI API pipeline (default model `gpt-4o-mini`).
* `prepare_training_data.py` — Builds training JSONL from labeled CSV (for local fine-tuning).
* `train_model.py` — Fine-tunes base model with LoRA (MLX-LM).
* `convert_chinese.py` — Simplified → Traditional conversion helper (OpenCC).
* `convert_data_format.py` — Converts labeled data into MLX-LM chat fine-tune format.
* `requirements.txt` — Minimal dependency list (you will install a few extras depending on the approach).

---

## 2) Output format (what you get)

Both pipelines write **JSONL**. Each line is a dictionary like:

```json
{
  "name": "程富",
  "rank": "行在监察御史",
  "action": "升为行在大理寺左少卿",
  "event_type": "Appointment",
  "time": "丁亥",
  "quote": "升行在监察御史程富为行在大理寺左少卿",
  "chunk_id": 1
}
```

Notes:

* `quote` should be a **verbatim snippet** supporting the extraction.
* `chunk_id` maps back to the text chunk used for inference (useful for debugging).
* The GPT pipeline may also include an internal `_reasoning` field depending on your prompt/settings (you can strip it later if you don’t want it saved).

---

## 3) Setup

### 3.1 Create environment & install base deps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3.2 Install extras per approach

#### If you run **Approach B (GPT-4o-mini)**

```bash
pip install openai httpx pydantic
```

#### If you run **Approach A (Local MLX LoRA)** on Apple Silicon

```bash
pip install mlx mlx-lm
```

#### If you run **Approach A (Ollama fallback)**

Install Ollama and pull a model you want to use (then configure `API_URL` / model name in `runNER.py`).

---

## 4) Approach A — Local model (MLX LoRA / Ollama)

### 4.1 Configure `runNER.py`

Open `runNER.py` and update the config block near the top:

* `PDF_PATH` — your input PDF path
* `OUTPUT_JSON` — output file name (default `cleaned_data.json`)
* `USE_MLX_MODEL` — `True` to use MLX LoRA, `False` to use Ollama
* If MLX:

  * `MODEL_NAME` — base model name (default: `Qwen/Qwen2.5-3B-Instruct-4bit`)
  * `ADAPTER_DIR` — LoRA adapter path (default: `adapters/ner_adapter`)
* If Ollama:

  * `API_URL` — default is `http://localhost:11434/api/generate`

Debug controls inside the file:

* `DEBUG_START_PAGE`, `DEBUG_PAGES` — which pages to process in debug mode

### 4.2 Run

Debug run (short, recommended first):

```bash
python runNER.py --debug
```

Full run:

```bash
python runNER.py
```

### 4.3 Output

* Writes to `OUTPUT_JSON` (default `cleaned_data.json`)
* Logs progress chunk-by-chunk

---

## 5) Approach B — GPT-4o-mini (OpenAI API)

### 5.1 Set API key

Recommended: environment variable

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

(You *can* hardcode it in the file, but env var is cleaner.)

Note: If you are in Hong Kong, you may need a VPN to access OpenAI services.

### 5.2 Configure `runNER_GPT.py`

Edit the config block near the top:

* `OPENAI_MODEL` — default: `gpt-4o-mini`
* `PDF_PATH` — input PDF path
* `OUTPUT_FILE` — default: `cleaned_data_gpt.jsonl`
* `MAX_WORKERS` — parallel requests (start with `1–2` if you hit instability)
* `CHUNK_SIZE`, `CHUNK_OVERLAP` — if you see truncated/invalid JSON, reduce `CHUNK_SIZE`
* `DEBUG_START_PAGE`, `DEBUG_PAGES` — debug page range

There are also pattern filters like:

* `EXCLUDE_TITLE_LIKE`
* `EXCLUDE_KINSHIP_ONLY`
* `EXCLUDE_SINGLE_CHAR_NAMES`
* `EXCLUDE_TOO_LONG_NAMES`

These are useful to suppress common false positives (titles-only, kinship-only mentions, etc.).

### 5.3 Run

Debug run:

```bash
python runNER_GPT.py --debug
```

Full run:

```bash
python runNER_GPT.py
```

### 5.4 Output

* Writes JSONL to `OUTPUT_FILE` (default `cleaned_data_gpt.jsonl`)

---

## 6) Training your own LoRA (for Approach A)

This section is only needed if you want to **fine-tune** the local model.

### 6.1 Prepare labeled data (`prepare_training_data.py`)

This script converts a labeled CSV into `train.jsonl`.

1. Open `prepare_training_data.py`
2. Set:

   * `INPUT_CSV` — your labeled CSV file
   * `OUTPUT_JSONL` — default `train.jsonl`

Your CSV is expected to contain at least:

* `chunk_id`
* `chunk_text`
* `gpt_ner_output` (or your gold output string)

Then run:

```bash
python prepare_training_data.py
```

### 6.2 Convert to MLX-LM format (`convert_data_format.py`)

1. Open `convert_data_format.py`
2. Set:

   * `INPUT_FILE` — typically `train.jsonl`
   * `OUTPUT_FILE` — default `train_mlx.jsonl`

Run:

```bash
python convert_data_format.py
```

### 6.3 (Optional) Convert Simplified → Traditional (`convert_chinese.py`)

1. Open `convert_chinese.py`
2. Set:

   * `INPUT_FILE`, `OUTPUT_FILE`

Run:

```bash
python convert_chinese.py
```

### 6.4 Train LoRA (`train_model.py`)

`train_model.py` trains LoRA adapters using MLX-LM. It expects training data at the path defined inside the script (`TRAIN_DATA`), so make sure that points to your `train_mlx.jsonl`.

Common runs:

Debug training (fast sanity check):

```bash
python train_model.py --debug
```

Regular training:

```bash
python train_model.py --iters 2000 --batch-size 2 --learning-rate 2e-5
```

Test adapter after training:

```bash
python train_model.py --skip-train --test
```

Fuse adapter into a standalone model (optional):

```bash
python train_model.py --skip-train --fuse
```

### 6.5 Use the adapter in `runNER.py`

Set in `runNER.py`:

* `USE_MLX_MODEL=True`
* `ADAPTER_DIR="adapters/ner_adapter"` (or the adapter path you trained to)

Run `runNER.py` again.

---

## 7) Troubleshooting

### 7.1 "Invalid JSON: EOF while parsing a string …" (GPT pipeline)

This means the model returned **truncated JSON** (output cut off mid-string). Typical fixes:

1. **Reduce** `CHUNK_SIZE` in `runNER_GPT.py` (first thing to try).
2. Reduce concurrency: set `MAX_WORKERS = 1`.
3. If your prompt allows verbose fields (like `_reasoning`), remove them to shrink output.
4. Add stronger JSON constraints (if you later revise the API call): schema / strict JSON-only formatting.

### 7.2 Pipeline feels slow / "stuck"

* Start with `--debug` to confirm the pipeline works end-to-end.
* For GPT: lower `MAX_WORKERS`.
* For Local MLX: ensure the base model and adapter are on fast storage; try fewer workers.

### 7.3 "Model outputs lots of titles, not names"

Enable/keep the exclusion filters in `runNER_GPT.py` (title-like / single-character / kinship-only), and tighten name validation rules.

---

## 8) Licence

### This project is conducted under the PolyU Department of Chinese History and Culture STEM Internship. Reference only. 
