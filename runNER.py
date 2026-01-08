import os
import re
import json
import logging
import argparse
import threading
import inspect
from typing import List, Dict, Any, Optional, Tuple, Iterator

import requests
from tqdm import tqdm

# --- Configuration ---
# Use fine-tuned MLX model instead of Ollama
USE_MLX_MODEL = True  # Set to False to use Ollama

# IMPORTANT:
# - MODEL_NAME should be the SAME base family used for LoRA training (architecture must match)
# - For memory stability on a 24GB Mac, prefer 4-bit MLX community models
MODEL_NAME = "mlx-community/Qwen2.5-3B-Instruct-4bit"   # Base model name/path
ADAPTER_PATH = "adapters"                               # LoRA adapter directory
API_URL = "http://127.0.0.1:11434/api/generate"         # Ollama endpoint (fallback)

# Paths
INPUT_FILE = "/Users/chunmanchan/Downloads/Alan/CHC_Intern/NER_Proj/NER(Dec.25)/source/明英宗实录（可检索版）.pdf"
OUTPUT_FILE = "cleaned_data.jsonl"

# Chunking (characters)
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

# Generation
MAX_TOKENS = 700
TEMPERATURE = 0.0  # deterministic intent; MLX uses sampler (not temp kw)
TOP_P = 1.0

# MLX KV cache cap (keeps memory stable)
# Make this comfortably larger than (system_prompt + chunk) tokens.
MLX_MAX_KV_SIZE = 2048

# Debug page range
DEBUG_PAGES = 50        # if --debug, how many pages to read
DEBUG_START_PAGE = 20    # if --debug, where to start

# Parallelism (Ollama only; MLX must be sequential)
MAX_WORKERS = 1

# Output artifacts
SAVE_CHUNKS_TO_DISK = True
MAX_SAVED_CHUNKS = 200
MAX_BAD_DUMPS = 200

# Clear MLX cache periodically (helps long runs)
CLEAR_MLX_CACHE_EVERY = 5

# Ollama options
OLLAMA_NUM_CTX = 4096
OLLAMA_KEEP_ALIVE = "30m"
OLLAMA_TIMEOUT = 900

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

SYSTEM_PROMPT = """
你是明朝历史资料专家。请从《明实录》片段中提取官员资讯。

【核心原则：基于证据 (Grounding)】
1. **严格对应**：资讯必须来自原文，不可编造或补全（如：原文写"尚书"不可补为"吏部尚书"）。
2. **禁止幻觉**：如果文本没提到官职，rank 必须填 null。

【重要：人名过滤规则 (Negative Constraints)】
1. **排除非人类**：不要提取「皇太子」、「皇帝」、「上」、「百官」、「文武群臣」、「作人」、「民业」、「军官」、「边将」、「贼寇」等泛指名词或称谓。只提取有**具体姓名**的人（如：于谦、张辅）。
2. **排除物品与地名**：不要将物品（如：珍禽异兽）或地名误认为人名。
3. **处理并列名单**：如果原文是「太监沐敬丰城侯李贤...」，这是两个人（沐敬、李贤），**绝对不要**把他们合并成一个名字。请拆分成多个 JSON 对象。

【输出栏位】
- "name": 官员姓名 (必须是具体人名，如 "张辅"，不能是 "太师"、"皇太子"、"百官"、"作人"、"民业"等)。
- "rank": 官职 (如 "太师"、"英国公")。
- "action": 事件摘要。
- "event_type": 事件分类 (Appointment/Military/Diplomacy/Disaster/Death/Other)。
- "time": 事件时间 (如 "宣德九年")。
- "_reasoning": (必要) 简短引用原文证明此人存在，并说明为何判定他是官员。

【范例 (Few-Shot)】
Input: "宣德九年，少师蹇义卒。"
Output: [
{
"name": "蹇义",
"rank": "少师",
"event_type": "Death",
"action": "去世",
"time": "宣德九年",
"_reasoning": "原文『少师蹇义卒』，蹇义为具体人名，提到蹇义及其官职少师，事件为去世。"
}
]

Input: "命行在工部尚书李友直提督供应柴炭。"
Output: [
{
"name": "李友直",
"rank": "行在工部尚书",
"event_type": "Appointment",
"action": "提督供应柴炭",
"time": null
"_reasoning": "原文『命行在工部尚书李友直...』，李友直被任命提督供应柴炭。",
}
]

【输出格式】
只输出标准 JSON List，不要包含 Markdown 标记或其他解释。
"""

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def find_good_cut(text: str, start: int, hard_end: int, lookback: int = 250) -> int:
    """
    Try to cut at punctuation/newline close to the chunk end.
    """
    look_start = max(start + 1, hard_end - lookback)
    candidate = text[look_start:hard_end]
    m = re.search(r"[。！？；\n]", candidate[::-1])
    if m:
        cut_back = m.start()
        return hard_end - cut_back
    return hard_end

def parse_json_list(response_text: str) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Returns (parsed_ok, items).
    parsed_ok=True with items=[] is VALID (means model returned empty list).
    """
    if not response_text:
        return (False, [])

    s = response_text.strip()

    # Remove fenced code blocks
    if "```" in s:
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1]
            s = re.sub(r"^\s*json\s*", "", s.strip(), flags=re.I)

    s = s.strip()

    # Extract first JSON array/object if there is extra text
    if not (s.startswith("[") or s.startswith("{")):
        m = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", s)
        if not m:
            return (False, [])
        s = m.group(1).strip()

    try:
        data = json.loads(s)
    except Exception:
        return (False, [])

    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return (False, [])

    out: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "official_name" in item and "name" not in item:
            item["name"] = item.pop("official_name")
        # keep empty list result as valid (out stays empty)
        if item.get("name") and str(item.get("name")) != "[]":
            out.append(item)

    return (True, out)

class MLXBackend:
    def __init__(self):
        self._lock = threading.Lock()

        # Validate adapter files early
        if ADAPTER_PATH:
            cfg = os.path.join(ADAPTER_PATH, "adapter_config.json")
            wts = os.path.join(ADAPTER_PATH, "adapters.safetensors")
            if not (os.path.isfile(cfg) and os.path.isfile(wts)):
                raise FileNotFoundError(
                    f"ADAPTER_PATH='{ADAPTER_PATH}' must contain adapter_config.json and adapters.safetensors"
                )

        logging.info("Loading MLX model once (base + adapter)...")
        logging.info(f"MODEL_NAME={MODEL_NAME}")
        logging.info(f"ADAPTER_PATH={ADAPTER_PATH}")

        from mlx_lm import load
        self.model, self.tokenizer = load(MODEL_NAME, adapter_path=ADAPTER_PATH)

        # Import stream_generate + sampler builder with version-safe fallbacks
        self.stream_generate = None
        self.make_sampler = None

        try:
            from mlx_lm import stream_generate
            self.stream_generate = stream_generate
        except Exception:
            from mlx_lm.utils import stream_generate
            self.stream_generate = stream_generate

        # make_sampler can live in different modules across versions
        for mod_path in [
            "mlx_lm.sample_utils",
            "mlx_lm.generate",
            "mlx_lm.utils",
        ]:
            try:
                mod = __import__(mod_path, fromlist=["make_sampler"])
                self.make_sampler = getattr(mod, "make_sampler", None)
                if self.make_sampler:
                    break
            except Exception:
                continue

        self._sig = inspect.signature(self.stream_generate)

    def _build_chat_prompt(self, user_text: str) -> str:
        """
        Qwen instruct models behave better with chat templates.
        Falls back to plain concatenation if tokenizer lacks chat template.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"【输入文本】:\n{user_text}\n\n【输出 JSON List】:"},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        # fallback
        return f"{SYSTEM_PROMPT}\n\n【输入文本】:\n{user_text}\n\n【输出 JSON List】:"

    def generate(self, chunk_text: str) -> str:
        import mlx.core as mx

        prompt = self._build_chat_prompt(chunk_text)

        # Build sampler if supported (NO 'temp' kw anywhere)
        sampler = None
        if self.make_sampler is not None:
            try:
                # temp=0.0 usually means greedy / deterministic intent
                sampler = self.make_sampler(temp=TEMPERATURE, top_p=TOP_P)
            except Exception:
                sampler = None

        kwargs = {"max_tokens": MAX_TOKENS}

        if "sampler" in self._sig.parameters and sampler is not None:
            kwargs["sampler"] = sampler
        if "max_kv_size" in self._sig.parameters and MLX_MAX_KV_SIZE:
            kwargs["max_kv_size"] = MLX_MAX_KV_SIZE

        # Serialize MLX calls for stability
        with self._lock:
            pieces: List[str] = []
            for resp in self.stream_generate(self.model, self.tokenizer, prompt, **kwargs):
                # resp.text is the last emitted segment
                pieces.append(getattr(resp, "text", "") or "")
            out = "".join(pieces)

        # Make sure pending ops are evaluated, then optionally clear cache
        mx.eval()
        return out

    def clear_cache(self):
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass

class OllamaBackend:
    def __init__(self):
        r = requests.get("http://127.0.0.1:11434/api/version", timeout=3)
        r.raise_for_status()
        logging.info("Ollama server is running and responsive.")

    def generate(self, prompt: str) -> str:
        payload = {
            "model": MODEL_NAME,   # if you want a different ollama model name, split configs
            "prompt": prompt,
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": 0.0,
                "num_ctx": OLLAMA_NUM_CTX,
                "num_predict": MAX_TOKENS,
            }
        }
        resp = requests.post(API_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("response", "")

def iter_pdf_pages(input_path: str, start_page: int, max_pages: Optional[int]) -> Iterator[str]:
    import pypdf
    reader = pypdf.PdfReader(input_path)
    total = len(reader.pages)

    if start_page < 0:
        start_page = 0
    end_page = total if max_pages is None else min(total, start_page + max_pages)

    logging.info(f"Reading pages: {start_page} .. {end_page-1} (total pages: {total})")

    for i in range(start_page, end_page):
        t = reader.pages[i].extract_text()
        if t:
            yield t

def stream_chunks_from_pages(pages: Iterator[str]) -> Iterator[str]:
    """
    Build chunks incrementally from page text to avoid holding the full PDF in memory.
    """
    buffer = ""
    for page_text in pages:
        page_text = normalize_text(page_text)
        if not page_text:
            continue

        if buffer:
            buffer += "\n"
        buffer += page_text

        # emit chunks while buffer is large enough
        while len(buffer) >= CHUNK_SIZE:
            start = 0
            hard_end = min(start + CHUNK_SIZE, len(buffer))
            end = find_good_cut(buffer, start, hard_end)
            chunk = buffer[start:end].strip()
            if chunk:
                yield chunk
            # keep overlap
            buffer = buffer[max(0, end - CHUNK_OVERLAP):]

    # last remainder
    tail = buffer.strip()
    if tail:
        yield tail

class MingShiluNERPipeline:
    def __init__(self, debug_mode: bool):
        self.debug_mode = debug_mode
        self.write_lock = threading.Lock()

        self.output_dir = os.path.dirname(OUTPUT_FILE) or "."
        self.chunk_dir = os.path.join(self.output_dir, "Chunks")
        self.bad_chunk_dir = os.path.join(self.output_dir, "Bad_Chunks")
        ensure_dir(self.chunk_dir)
        ensure_dir(self.bad_chunk_dir)

        self.saved_chunks = 0
        self.bad_dumps = 0

        self.backend = MLXBackend() if USE_MLX_MODEL else OllamaBackend()

        logging.info(f"Output: {OUTPUT_FILE}")
        logging.info(f"Chunks dir: {self.chunk_dir} (SAVE_CHUNKS_TO_DISK={SAVE_CHUNKS_TO_DISK})")
        logging.info(f"Bad chunks dir: {self.bad_chunk_dir} (MAX_BAD_DUMPS={MAX_BAD_DUMPS})")
        logging.info("Backend: MLX (sequential, sampler-based)" if USE_MLX_MODEL else "Backend: Ollama")

    def dump_bad(self, chunk_id: int, chunk_text: str, raw_text: str):
        if self.bad_dumps >= MAX_BAD_DUMPS:
            return
        path = os.path.join(self.bad_chunk_dir, f"bad_chunk_{chunk_id}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("=== CHUNK TEXT ===\n")
            f.write(chunk_text)
            f.write("\n\n=== RAW MODEL RESPONSE ===\n")
            f.write(raw_text)
        self.bad_dumps += 1

    def save_chunk(self, chunk_id: int, chunk_text: str):
        if not SAVE_CHUNKS_TO_DISK:
            return
        if self.saved_chunks >= MAX_SAVED_CHUNKS:
            return
        path = os.path.join(self.chunk_dir, f"chunk_{chunk_id}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(chunk_text)
        self.saved_chunks += 1

    def run(self):
        # reset output (JSONL)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            pass

        if self.debug_mode:
            start_page = DEBUG_START_PAGE
            max_pages = DEBUG_PAGES
        else:
            start_page = 0
            max_pages = None

        pages = iter_pdf_pages(INPUT_FILE, start_page=start_page, max_pages=max_pages)
        chunk_iter = stream_chunks_from_pages(pages)

        total_entities = 0
        chunk_id = 0

        # Open output once for performance
        with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
            for chunk_text in tqdm(chunk_iter, desc="Processing Chunks"):
                self.save_chunk(chunk_id, chunk_text)

                try:
                    if USE_MLX_MODEL:
                        raw = self.backend.generate(chunk_text)
                    else:
                        # Ollama wants a full prompt string
                        prompt = f"{SYSTEM_PROMPT}\n\n【输入文本】:\n{chunk_text}\n\n【输出 JSON List】:"
                        raw = self.backend.generate(prompt)
                except Exception as e:
                    self.dump_bad(chunk_id, chunk_text, f"[EXCEPTION] {e}")
                    chunk_id += 1
                    continue

                parsed_ok, items = parse_json_list(raw)
                if not parsed_ok:
                    self.dump_bad(chunk_id, chunk_text, raw)
                    chunk_id += 1
                    continue

                # parsed_ok=True, items may be empty => valid
                if items:
                    for it in items:
                        it["chunk_id"] = chunk_id
                        out_f.write(json.dumps(it, ensure_ascii=False) + "\n")
                    total_entities += len(items)

                # Periodic MLX cache clear (helps prevent memory pressure)
                if USE_MLX_MODEL and (chunk_id + 1) % CLEAR_MLX_CACHE_EVERY == 0:
                    self.backend.clear_cache()

                chunk_id += 1

        logging.info(f"Done. Total extracted entities: {total_entities}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Debug mode: use DEBUG_START_PAGE + DEBUG_PAGES")
    args = parser.parse_args()

    print("模式：提取所有实体 (Full Extraction)")
    MingShiluNERPipeline(debug_mode=args.debug).run()
