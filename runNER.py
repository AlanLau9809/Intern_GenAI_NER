import requests
import json
import re
import time
import logging
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# --- Configuration ---
MODEL_NAME = "qwen2.5:7b" 
API_URL = "http://localhost:11434/api/generate"

# 請確認你的檔案路徑
INPUT_FILE = "/Users/chunmanchan/Downloads/Alan/CHC_Intern/NER_Proj/NER(Dec.25)/source/明英宗实录（可检索版）.pdf"
OUTPUT_FILE = "ming_shilu_ner_result.jsonl" 

# 並行處理數量
MAX_WORKERS = 3

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(threadName)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

class MingShiluNERPipeline:
    def __init__(self, model_name: str, api_url: str):
        self.model_name = model_name
        self.api_url = api_url
        self.lock = threading.Lock() 

        # --- UPDATED PROMPT: 加入「證據歸因 (Grounding)」 ---
        self.system_prompt = """
你是明朝歷史資料結構化專家。你的任務是「原樣提取」文本中的資訊。

【嚴格限制】
1. **只能提取**文本中「明確寫出」的內容。
2. **禁止**使用你自己的歷史知識來補全官職或事件。
3. 如果文本只寫了「尚書」，你就只能填「尚書」，不能自動補成「吏部尚書」。
4. 如果文本沒提到官職，rank 必須填 null。

【提取目標】
請將文本中出現的每一位官員轉換為一個 JSON 對象：
1. "name": 官員姓名。
2. "rank": 官職。若未提及則填 null。
3. "action": 事件摘要。
4. "quote": **證據原文**。請直接複製文本中證明該官員和官職的那句話，不要改動任何字。

【輸出規則】
- 輸出必須是標準的 JSON List。
- 如果無資料，回傳 []。
"""

    def _clean_json_response(self, response_text: str) -> List[Dict[str, Any]]:
        """清理 LLM 的輸出"""
        try:
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            data = json.loads(response_text)
            
            if isinstance(data, dict): 
                data = [data]
            
            if not isinstance(data, list):
                return []

            valid_items = []
            for item in data:
                if not isinstance(item, dict): continue
                
                # 相容性處理
                if 'official_name' in item and 'name' not in item:
                    item['name'] = item.pop('official_name')

                # 資料清洗：如果 quote 為空，通常代表是幻覺，標記警告
                # 這裡我們先只保留有名字的
                if item.get("name") and str(item.get("name")) != "[]":
                    valid_items.append(item)
            
            return valid_items

        except json.JSONDecodeError:
            return []

    def process_single_chunk(self, chunk_data: Dict) -> List[Dict]:
        """Worker 函數"""
        chunk_id = chunk_data['id']
        text = chunk_data['text']
        
        full_prompt = f"{self.system_prompt}\n\n【輸入文本】:\n{text}\n\n【輸出 JSON List】:"
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.0, # 設為 0，完全禁止創造性
                "num_ctx": 2048
            }
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=300)
            if response.status_code == 200:
                raw_text = response.json().get('response', '')
                extracted = self._clean_json_response(raw_text)
                
                for item in extracted:
                    item['chunk_id'] = chunk_id
                
                return extracted
            else:
                logging.error(f"Chunk {chunk_id} API Error: {response.status_code}")
                return []
        except Exception as e:
            logging.error(f"Chunk {chunk_id} Failed: {e}")
            return []

    def read_pdf(self, input_path: str, debug_mode: bool) -> str:
        """讀取 PDF"""
        try:
            import pypdf
            reader = pypdf.PdfReader(input_path)
            total_pages = len(reader.pages)
            
            if debug_mode:
                logging.info(f"DEBUG 模式開啟：只處理前 10 頁 (總頁數: {total_pages})")
                pages_to_read = reader.pages[:10]
            else:
                pages_to_read = reader.pages

            text_content = []
            for i, page in enumerate(pages_to_read):
                content = page.extract_text()
                if content:
                    text_content.append(content)
            
            logging.info(f"PDF 讀取完畢，共處理 {len(pages_to_read)} 頁")
            return "\n".join(text_content)
        except Exception as e:
            logging.error(f"PDF 讀取失敗: {e}")
            return ""

    def make_chunks(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[Dict]:
        chunks = []
        start = 0
        chunk_counter = 0
        text = text.replace("\n", "") 

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append({
                "id": chunk_counter,
                "text": chunk_text
            })
            chunk_counter += 1
            start += (chunk_size - overlap)
            
        logging.info(f"文本已切分為 {len(chunks)} 個區塊 (Size: {chunk_size}, Overlap: {overlap})")
        return chunks

    def run(self, input_path: str, output_path: str, debug_mode: bool):
        raw_text = self.read_pdf(input_path, debug_mode)
        if not raw_text: return
        
        chunks = self.make_chunks(raw_text)
        
        logging.info(f"開始並行處理 (Workers: {MAX_WORKERS})...")
        logging.info(f"結果將即時寫入: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            pass

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_chunk = {executor.submit(self.process_single_chunk, chunk): chunk for chunk in chunks}
            
            processed_count = 0
            total_entities = 0

            for future in as_completed(future_to_chunk):
                processed_count += 1
                result_data = future.result()
                
                if result_data:
                    total_entities += len(result_data)
                    with self.lock:
                        with open(output_path, 'a', encoding='utf-8') as f:
                            for item in result_data:
                                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    
                    first_person = result_data[0].get('name', 'Unknown')
                    logging.info(f"[{processed_count}/{len(chunks)}] 提取 {len(result_data)} 筆 (如: {first_person})")
                else:
                    if processed_count % 10 == 0:
                        logging.info(f"[{processed_count}/{len(chunks)}] 無資料或處理完畢")

        logging.info(f"任務完成！共提取 {total_entities} 個實體。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Debug mode: Process only first 10 pages")
    args = parser.parse_args()

    pipeline = MingShiluNERPipeline(MODEL_NAME, API_URL)
    pipeline.run(INPUT_FILE, OUTPUT_FILE, args.debug)