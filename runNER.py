import requests
import json
import re
import time
import logging
import subprocess
import shutil
import argparse
from typing import List, Dict, Any, Tuple

# --- Configuration ---
MODEL_NAME = "deepseek-r1:14b"
API_URL = "http://localhost:11434/api/generate"
# Update this path to your actual file location
INPUT_FILE = "/Users/chunmanchan/Downloads/Alan/CHC_Intern/NER_Proj/NER(Dec.25)/source/明英宗实录（可检索版）.pdf"
OUTPUT_FILE = "semantic_cleaned_data.json"

# Semantic Search Configuration
# This model is excellent for Chinese/English semantic matching and runs fast on CPU
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
SIMILARITY_THRESHOLD = 0.35  # 0.0 to 1.0. Lower = more loose matches. 0.35 is good for short text.

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

class MingShiluExtractor:
    def __init__(self, model_name: str, api_url: str, keyword: str = None, use_semantic: bool = False):
        self.model_name = model_name
        self.api_url = api_url
        self.keyword = keyword
        self.use_semantic = use_semantic
        self.embedding_model = None

        # Lazy load embedding model only if needed
        if self.use_semantic and self.keyword:
            try:
                from sentence_transformers import SentenceTransformer
                logging.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            except ImportError:
                logging.error("sentence-transformers not installed. Run: pip install sentence-transformers")
                raise

        # System Prompt
        self.system_prompt = """
You are a historian specializing in the Ming Dynasty. 
Your task is to perform Named Entity Recognition (NER) on the provided text.

Extract the following entities into a strict JSON list format:
1. "official_name": The name of the official.
2. "official_rank": The title or rank.
3. "action": Briefly describe the action or event.
4. "time_context": Extract the year/month mentioned.
5. "relevance": Why is this text relevant to the user's keyword?

Rules:
- Output ONLY a valid JSON list. 
- Handle ancient text segmentation (斷句) internally.
"""

    def _filter_chunks_semantically(self, chunks: List[str]) -> List[str]:
        """
        Uses vector embeddings to find chunks conceptually similar to the keyword.
        """
        if not self.embedding_model or not self.keyword:
            return chunks

        from sentence_transformers import util
        
        logging.info(f"Encoding {len(chunks)} chunks for semantic search...")
        
        # 1. Encode the keyword
        query_embedding = self.embedding_model.encode(self.keyword, convert_to_tensor=True)
        
        # 2. Encode all chunks (Batch processing)
        chunk_embeddings = self.embedding_model.encode(chunks, convert_to_tensor=True)
        
        # 3. Calculate Cosine Similarity
        # Returns a list of scores (0.0 to 1.0)
        cosine_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]

        # 4. Filter
        relevant_chunks = []
        
        # Combine chunk and score, then sort by score descending
        scored_chunks = []
        for i, score in enumerate(cosine_scores):
            if score > SIMILARITY_THRESHOLD:
                scored_chunks.append((score.item(), chunks[i]))

        # Sort: Highest similarity first
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        logging.info(f"--- Semantic Search Results for '{self.keyword}' ---")
        for score, chunk in scored_chunks[:5]: # Log top 5 matches
            snippet = chunk.replace('\n', '')[:50]
            logging.info(f"Score: {score:.4f} | Text: {snippet}...")

        # Return only the text parts
        return [item[1] for item in scored_chunks]

    def _clean_json_response(self, response_text: str) -> List[Dict[str, Any]]:
        # (Same robust cleaning logic as before)
        data = None
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            try:
                match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
                if match:
                    data = json.loads(match.group(1))
                else:
                    match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    if match:
                        data = json.loads(match.group(0))
            except json.JSONDecodeError:
                return []
        
        if isinstance(data, dict): return [data]
        if isinstance(data, list): return [item for item in data if isinstance(item, dict)]
        return []

    def query_llm(self, text_chunk: str, retries: int = 2) -> List[Dict[str, Any]]:
        # Construct Prompt
        prompt = f"{self.system_prompt}\n\nUser Keyword Focus: {self.keyword}\n\nInput:\n{text_chunk}\n\nOutput:"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_ctx": 4096}
        }

        for attempt in range(retries):
            try:
                response = requests.post(self.api_url, json=payload, timeout=300)
                response.raise_for_status()
                return self._clean_json_response(response.json().get('response', ''))
            except Exception as e:
                logging.warning(f"Attempt {attempt+1} failed: {e}")
                time.sleep(1)
        return []

    def chunk_text(self, text: str, max_chars: int = 800) -> List[str]:
        # Simple overlap chunking to ensure context isn't lost at cut-off points
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + max_chars, text_len)
            chunks.append(text[start:end])
            start += (max_chars - 100) # 100 char overlap
        return chunks

    def run(self, input_path: str, output_path: str, debug_mode: bool = False):
        logging.info(f"Starting pipeline. Semantic Search: {self.use_semantic}")
        
        # 1. Load Data
        text = ""
        try:
            if input_path.lower().endswith('.pdf'):
                import pypdf
                reader = pypdf.PdfReader(input_path)
                text = "\n".join([page.extract_text() or "" for page in reader.pages])
            else:
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read()
        except Exception as e:
            logging.error(f"Error reading file: {e}")
            return

        # 2. Chunk Data
        chunks = self.chunk_text(text)
        logging.info(f"Total raw chunks: {len(chunks)}")
        
        # 3. Semantic Filtering (The Approach A Logic)
        if self.keyword:
            if self.use_semantic:
                chunks = self._filter_chunks_semantically(chunks)
            else:
                # Fallback to simple string matching
                chunks = [c for c in chunks if self.keyword in c]
            
            logging.info(f"Chunks remaining after filtering: {len(chunks)}")

        if not chunks:
            logging.warning("No relevant chunks found. Exiting.")
            return

        if debug_mode:
            chunks = chunks[:5]

        # 4. LLM Processing
        all_data = []
        for i, chunk in enumerate(chunks):
            logging.info(f"Analyzing Chunk {i+1}/{len(chunks)} with GenAI...")
            extracted = self.query_llm(chunk)
            if extracted:
                for item in extracted:
                    item['source_chunk_id'] = i
                all_data.extend(extracted)

        # 5. Save
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        logging.info(f"Done. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--keyword", type=str, help="Search keyword")
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic search (use exact match only)")
    args = parser.parse_args()

    target_keyword = args.keyword
    if not target_keyword:
        target_keyword = input("Enter keyword (e.g., 地震, 兵變, 于謙): ").strip()

    pipeline = MingShiluExtractor(
        model_name=MODEL_NAME, 
        api_url=API_URL, 
        keyword=target_keyword,
        use_semantic=not args.no_semantic # Default to True
    )
    
    pipeline.run(INPUT_FILE, OUTPUT_FILE, debug_mode=args.debug)