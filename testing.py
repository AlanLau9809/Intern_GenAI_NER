import re
import fitz  # PyMuPDF
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

def extract_pdf_text(pdf_path: str) -> Dict[str, Any]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        pages.append({"page": i, "text": text})
    return {"doc_id": pdf_path, "pages": pages}

def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[“”]", '"', s)
    s = re.sub(r"[’]", "'", s)
    return s

def validate_entity(entity: Dict[str, Any], pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    page_num = entity.get("page")
    evidence = entity.get("evidence") or entity.get("text") or ""
    if not page_num or page_num < 1 or page_num > len(pages):
        return {**entity, "status": "rejected", "reason": "invalid page"}

    page_text = pages[page_num - 1]["text"]

    # 1) exact match
    idx = page_text.find(evidence)
    if idx != -1:
        return {**entity, "status": "accepted", "start": idx, "end": idx + len(evidence)}

    # 2) normalized match
    n_page = normalize(page_text)
    n_evid = normalize(evidence)
    n_idx = n_page.find(n_evid)
    if n_idx != -1:
        return {**entity, "status": "accepted", "match": "normalized"}

    return {**entity, "status": "rejected", "reason": "not found"}

# --- placeholder for LLM extraction ---
def llm_extract_entities(pages: List[Dict[str, Any]], entity_types: List[str]) -> Dict[str, Any]:
    # You would:
    # - chunk pages
    # - pass to LLM with strict JSON schema
    # - parse JSON
    return {"entities": []}

def run_pipeline(pdf_path: str, entity_types: List[str]) -> Dict[str, Any]:
    doc_store = extract_pdf_text(pdf_path)                 # Step 1
    # Step 2: store doc_store (JSON/DB). Skipped here.

    llm_result = llm_extract_entities(doc_store["pages"], entity_types)  # Step 3

    validated = [validate_entity(e, doc_store["pages"]) for e in llm_result.get("entities", [])]  # Step 4
    accepted = [e for e in validated if e["status"] == "accepted"]
    rejected = [e for e in validated if e["status"] == "rejected"]

    return {"doc_id": doc_store["doc_id"], "accepted": accepted, "rejected": rejected}  # Step 5
