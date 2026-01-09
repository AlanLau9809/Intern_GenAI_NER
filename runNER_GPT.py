import os
import re
import json
import time
import logging
import argparse
from typing import List, Dict, Any, Optional, Iterator, Literal, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from tqdm import tqdm
from openai import OpenAI, DefaultHttpxClient
from pydantic import BaseModel, Field, ConfigDict, ValidationError

# ============================================================
# OpenAI config
# ============================================================
USE_OPENAI = True
OPENAI_API_KEY = ""                 # <-- Put API Key here (or use env OPENAI_API_KEY)
OPENAI_MODEL = "gpt-4o-mini"        # gpt-5-mini may require org verification
OPENAI_MODEL_FALLBACKS = ["gpt-4o-mini", "gpt-4o"]  # tried if model not available

OPENAI_TEMPERATURE = 0
OPENAI_MAX_OUTPUT_TOKENS = 900      # keep moderate; we force short outputs

# Parallelize calls inside each chunk (biggest speedup if you keep segmentation)
MAX_WORKERS = 8

# Hard timeout to avoid “stuck forever”
CONNECT_TIMEOUT = 15.0
READ_TIMEOUT = 60.0
WRITE_TIMEOUT = 30.0

# Retry policy (SDK already retries some errors by default; you can tune it) :contentReference[oaicite:2]{index=2}
MAX_RETRIES = 2

# ============================================================
# Paths
# ============================================================
INPUT_FILE = "/Users/chunmanchan/Downloads/Alan/CHC_Intern/NER_Proj/NER(Dec.25)/source/明英宗实录（可检索版）.pdf"
OUTPUT_FILE = "cleaned_data_gpt.jsonl"

# ============================================================
# Chunking (characters)
# ============================================================
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 80

# Debug page range
DEBUG_PAGES = 10
DEBUG_START_PAGE = 20

# ============================================================
# Segmentation knobs
# ============================================================
SPLIT_ON_MARKERS = False
SPLIT_MARKERS = ["○"]
MARKERS_PER_REQUEST = 6   # ↑ bigger => fewer API calls (faster)

# ============================================================
# Accuracy knobs (all OPTIONAL)
# ============================================================
# If True, filter out title-like entities (e.g., "...皇帝", "...皇后") that are not “personal names”.
EXCLUDE_TITLE_LIKE = False

TITLE_LIKE_RE = re.compile(
    r"(皇帝|皇后|太子|皇太子|公主|国王|王爷|王|后|帝)$"
)

# OPTIONAL: handle kinship pattern like “X孙Y” => name should be Y
HANDLE_KINSHIP_X_SUN_Y = True
KINSHIP_SUN_RE = re.compile(r"^(.{1,6})孙(.{1,6})$")

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# ============================================================
# Prompt (general, but enforces quote grounding + no duplicates)
# ============================================================
SYSTEM_PROMPT = """
你是中文文本信息抽取专家。请从输入片段中提取“人物实体（具体姓名/专名）”及其相关事件。

【核心原则：基于证据】
1) 严格对应：所有信息必须来自原文，不可编造或补全。
2) 不确定就留空：原文没出现的字段，rank/time 填 null。
3) 必须给证据：quote 必须是原文中连续子串（5~40字），不可改写。

【姓名要求】
- name 必须是具体姓名/专名；不要输出代词或泛指（如：某人、其人、他、她、众人、诸官等）。
- 不要把多个人合并成一个 name。并列名单要拆成多条记录。
- 若只有称谓/头衔但无明确姓名，不要输出该条。
- 同一输入片段里：相同的人物+同一事件不要重复输出。

【字段说明】
- name: 人物姓名（具体姓名/专名）
- rank: 头衔/官职/身份（原文出现；没有则 null）
- action: 事件摘要（尽量短，但要能看出发生了什么；来自原文）
- event_type: Appointment/Military/Diplomacy/Disaster/Death/Other
- time: 时间（原文出现；没有则 null）
- quote: 原文短引（5~40字，必须在输入文本中出现）

【输出格式】
输出 JSON，顶层对象包含字段 items：{"items":[...]}。不要输出 Markdown 或额外解释。
"""

#- _reasoning: 1句说明（尽量短），解释为什么抽取（必须引用 quote 中的人名，最多 40 字）

# ============================================================
# Structured Output schema
# ============================================================
EventType = Literal["Appointment", "Military", "Diplomacy", "Disaster", "Death", "Other"]

class NERItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    rank: Optional[str] = None
    action: str
    event_type: EventType
    time: Optional[str] = None
    quote: str

    # reasoning: str = Field(..., alias="_reasoning")

class NEROutput(BaseModel):
    items: List[NERItem] = Field(default_factory=list)

# ============================================================
# Helpers
# ============================================================
def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def find_good_cut(text: str, start: int, hard_end: int, lookback: int = 250) -> int:
    look_start = max(start + 1, hard_end - lookback)
    candidate = text[look_start:hard_end]
    m = re.search(r"[。！？；\n]", candidate[::-1])
    if m:
        cut_back = m.start()
        return hard_end - cut_back
    return hard_end

def iter_pdf_pages(input_path: str, start_page: int, max_pages: Optional[int]) -> Iterator[str]:
    import pypdf
    reader = pypdf.PdfReader(input_path)
    total = len(reader.pages)

    start_page = max(0, start_page)
    end_page = total if max_pages is None else min(total, start_page + max_pages)
    logging.info(f"Reading pages: {start_page} .. {end_page-1} (total pages: {total})")

    for i in range(start_page, end_page):
        t = reader.pages[i].extract_text()
        if t:
            yield t

def stream_chunks_from_pages(pages: Iterator[str], chunk_size: int, overlap: int) -> Iterator[str]:
    buffer = ""
    for page_text in pages:
        page_text = normalize_text(page_text)
        if not page_text:
            continue

        buffer = f"{buffer}\n{page_text}" if buffer else page_text

        while len(buffer) >= chunk_size:
            start = 0
            hard_end = min(start + chunk_size, len(buffer))
            end = find_good_cut(buffer, start, hard_end)
            chunk = buffer[start:end].strip()
            if chunk:
                yield chunk
            buffer = buffer[max(0, end - overlap):]

    tail = buffer.strip()
    if tail:
        yield tail

def marker_segments(text: str) -> List[str]:
    """
    Segment by entries starting with markers (e.g. "○") and group N per call.
    """
    t = normalize_text(text)
    if not SPLIT_ON_MARKERS or not SPLIT_MARKERS:
        return [t]
    if not any(m in t for m in SPLIT_MARKERS):
        return [t]

    for m in SPLIT_MARKERS:
        t = re.sub(rf"\s*{re.escape(m)}\s*", f"\n{m}", t)
    t = t.strip()

    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]

    head: List[str] = []
    items: List[str] = []
    for ln in lines:
        if any(ln.startswith(m) for m in SPLIT_MARKERS):
            items.append(ln)
        else:
            head.append(ln)

    if not items:
        return [t]

    header = "\n".join(head).strip()
    segs: List[str] = []
    for i in range(0, len(items), MARKERS_PER_REQUEST):
        grp = "\n".join(items[i:i + MARKERS_PER_REQUEST])
        seg = f"{header}\n{grp}".strip() if header else grp
        segs.append(seg)
    return segs

def normalize_item(it: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    name = str(it.get("name", "")).strip()
    if not name:
        return None

    # Optional: drop title-like “names”
    if EXCLUDE_TITLE_LIKE and TITLE_LIKE_RE.search(name):
        return None

    # Optional: handle “X孙Y”
    if HANDLE_KINSHIP_X_SUN_Y:
        m = KINSHIP_SUN_RE.match(name)
        if m:
            name = m.group(2).strip()
            if not name:
                return None
            it["name"] = name

    # Normalize empty rank/time
    if it.get("rank") in ("", "null"):
        it["rank"] = None
    if it.get("time") in ("", "null"):
        it["time"] = None

    # quote must exist and be short-ish
    quote = str(it.get("quote", "")).strip()
    if not quote or len(quote) < 5 or len(quote) > 60:
        return None

    # keep reasoning short
    rs = str(it.get("_reasoning", "")).strip()
    if rs and len(rs) > 120:
        it["_reasoning"] = rs[:120]

    it["name"] = name
    return it

def dedup_key(it: Dict[str, Any]) -> Tuple[str, str, str, str, str, str]:
    return (
        str(it.get("name") or "").strip(),
        str(it.get("rank") or "").strip(),
        str(it.get("action") or "").strip(),
        str(it.get("event_type") or "").strip(),
        str(it.get("time") or "").strip(),
        str(it.get("quote") or "").strip(),
    )

# ============================================================
# OpenAI backend
# ============================================================
class OpenAIBackend:
    def __init__(self):
        key = OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("Missing OPENAI_API_KEY (set in code or env).")

        timeout = httpx.Timeout(
            READ_TIMEOUT,
            connect=CONNECT_TIMEOUT,
            write=WRITE_TIMEOUT,
            read=READ_TIMEOUT,
        )
        http_client = DefaultHttpxClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )

        self.client = OpenAI(
            api_key=key,
            timeout=timeout,         # supported by openai-python :contentReference[oaicite:3]{index=3}
            max_retries=MAX_RETRIES, # supported by openai-python :contentReference[oaicite:4]{index=4}
            http_client=http_client,
        )

        logging.info(f"OpenAI client ready. model={OPENAI_MODEL}, connect_timeout={CONNECT_TIMEOUT}s")

    def _call_once(self, model: str, text: str) -> NEROutput:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"【输入文本】\n{text}\n"},
        ]
        resp = self.client.responses.parse(
            model=model,
            input=messages,
            text_format=NEROutput,
            temperature=OPENAI_TEMPERATURE,
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        )
        parsed = getattr(resp, "output_parsed", None)
        if parsed is None:
            # As a fallback, try to parse output_text
            raw_text = (getattr(resp, "output_text", "") or "").strip()
            if not raw_text:
                return NEROutput(items=[])
            data = json.loads(raw_text)
            return NEROutput(**data) if isinstance(data, dict) else NEROutput(items=[])
        return parsed

    def extract(self, text: str) -> List[Dict[str, Any]]:
        # Try primary + fallbacks (handles “model not available” cases cleanly)
        models_to_try = [OPENAI_MODEL] + [m for m in OPENAI_MODEL_FALLBACKS if m != OPENAI_MODEL]
        last_err: Optional[Exception] = None

        for model in models_to_try:
            for attempt in range(1, 3 + 1):
                try:
                    parsed = self._call_once(model, text)
                    return [it.model_dump(by_alias=True) for it in parsed.items]
                except (ValidationError, json.JSONDecodeError) as e:
                    # often caused by too-long outputs / truncated JSON; retry once
                    last_err = e
                    time.sleep(0.6 * attempt)
                except Exception as e:
                    last_err = e
                    time.sleep(0.6 * attempt)

        raise last_err  # type: ignore

# ============================================================
# Pipeline
# ============================================================
class NERPipeline:
    def __init__(self, debug_mode: bool):
        self.debug_mode = debug_mode
        self.backend = OpenAIBackend()

        self.seen_global = set()  # global dedup across chunks

        logging.info(f"Output: {OUTPUT_FILE}")

    def run(self):
        # reset output
        with open(OUTPUT_FILE, "w", encoding="utf-8"):
            pass

        start_page = DEBUG_START_PAGE if self.debug_mode else 0
        max_pages = DEBUG_PAGES if self.debug_mode else None

        pages = iter_pdf_pages(INPUT_FILE, start_page=start_page, max_pages=max_pages)
        chunk_iter = stream_chunks_from_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)

        total_items = 0
        chunk_id = 0

        with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
            for chunk_text in tqdm(chunk_iter, desc="Processing Chunks"):
                segments = marker_segments(chunk_text)

                merged: List[Dict[str, Any]] = []
                # Parallel calls within the same chunk (big speedup)
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                    futs = {ex.submit(self.backend.extract, seg): seg for seg in segments}
                    for fut in as_completed(futs):
                        seg = futs[fut]
                        try:
                            items = fut.result()
                        except Exception as e:
                            logging.warning(f"Segment failed: {e}")
                            continue

                        # validate + normalize each item
                        for it in items:
                            it2 = normalize_item(it)
                            if not it2:
                                continue
                            # quote must be present in the segment text (hard grounding)
                            if it2["quote"] not in seg:
                                continue
                            merged.append(it2)

                # De-dup (within chunk + across chunks)
                out_items: List[Dict[str, Any]] = []
                seen_local = set()
                for it in merged:
                    k = dedup_key(it)
                    if k in seen_local or k in self.seen_global:
                        continue
                    seen_local.add(k)
                    self.seen_global.add(k)

                    it["chunk_id"] = chunk_id
                    out_items.append(it)

                for it in out_items:
                    out_f.write(json.dumps(it, ensure_ascii=False) + "\n")
                total_items += len(out_items)
                chunk_id += 1

        logging.info(f"Done. Total extracted items: {total_items}")

# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Debug mode: use DEBUG_START_PAGE + DEBUG_PAGES")
    args = parser.parse_args()

    print("模式：提取所有实体 (Full Extraction)")
    NERPipeline(debug_mode=args.debug).run()
