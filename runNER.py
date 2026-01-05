import requests
import json
import re
import time
import logging
import argparse
import threading
import os # Added for directory operations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "qwen2.5:3b" 
API_URL = "http://127.0.0.1:11434/api/generate"

# 請確認你的檔案路徑
INPUT_FILE = "/Users/chunmanchan/Downloads/Alan/CHC_Intern/NER_Proj/NER(Dec.25)/source/明英宗实录（可检索版）.pdf"
OUTPUT_FILE = "cleaned_data.json" 

# 並行處理數量 
MAX_WORKERS = 4

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s', 
    handlers=[logging.StreamHandler()] 
)

class MingShiluNERPipeline:
    def __init__(self, model_name: str, api_url: str): 
        self.model_name = model_name
        self.api_url = api_url
        self.lock = threading.Lock() 

        # --- 動態生成 Prompt ---
        base_instruction = """
你是明朝歷史資料結構化專家。請從《明實錄》片段中提取資訊。

【核心原則：基於證據 (Grounding)】
1. 你的任務類似於「模版標註」，必須嚴格對應原文。
2. **禁止幻覺**：如果原文只寫「尚書」，rank 就填「尚書」，不可補全為「吏部尚書」。
3. **原文引述**：必須提供 quote 欄位證明你的提取是正確的。
"""

        # Always use the "提取所有實體" logic
        filter_instruction = """
4. **提取目標**：提取文本中出現的**所有**官員及其相關事件。
"""

        task_instruction = """
【任務一：實體提取 (NER)】
提取所有官員的：
- name: 姓名
- rank: 官職 (未提及填 null)

【任務二：事件分類 (Event Classification)】
根據框架，將事件歸類為以下之一：
- "Appointment": 人事任免 (陞、調、黜、罷)
- "Military": 軍事衝突 (攻、戰、禦、平亂)
- "Diplomacy": 外交朝貢 (使、貢、撫)
- "Disaster": 災異救濟 (災、賑、免稅)
- "Death": 官員去世 (卒、死、逝)
- "Other": 其他

【範例 (Few-Shot)】
Input: "宣德九年，少師蹇義卒。"
Output: [{"name": "蹇義", "rank": "少師", "event_type": "Death", "action": "去世", "quote": "少師蹇義卒", "time": "宣德九年"}]

Input: "命行在工部尚書李友直提督供應柴炭。"
Output: [{"name": "李友直", "rank": "行在工部尚書", "event_type": "Appointment", "action": "提督供應柴炭", "quote": "命行在工部尚書李友直提督供應柴炭", "time": null}]

【輸出格式】
請輸出標準 JSON List。
"""
        # 組合 Prompt
        self.system_prompt = base_instruction + filter_instruction + task_instruction

    def _clean_json_response(self, response_text: str) -> List[Dict[str, Any]]:
        """清理 LLM 的輸出"""
        try:
            # 原本的去 code fence
            if "```json" in response_text:
                response_text = response_text.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in response_text:
                response_text = response_text.split("```", 1)[1].split("```", 1)[0]

            response_text = response_text.strip()

            # 新增：嘗試抓出第一個 JSON array/object
            if not response_text.startswith("[") and not response_text.startswith("{"):
                m = re.search(r"(\[.*\]|\{.*\})", response_text, re.S)
                if m:
                    response_text = m.group(1)

            data = json.loads(response_text)

            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                return []

            valid_items = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                if "official_name" in item and "name" not in item:
                    item["name"] = item.pop("official_name")

                if item.get("name") and str(item.get("name")) != "[]":
                    valid_items.append(item)

            return valid_items

        except Exception:
            return []

    def _check_ollama(self):
        """檢查 Ollama server 是否正常運行"""
        try:
            r = requests.get("http://127.0.0.1:11434/api/version", timeout=3)
            r.raise_for_status()
            logging.info("Ollama server is running and responsive.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Ollama server check failed: {e}. Please ensure Ollama is running.")
            raise

    def _dump_bad_response(self, chunk_id: int, chunk_text: str, raw_text: str):
        """將解析失敗的回應存檔，方便除錯"""
        bad_chunk_dir = os.path.join(os.path.dirname(OUTPUT_FILE), "Bad_Chunks")
        os.makedirs(bad_chunk_dir, exist_ok=True) # Ensure directory exists
        with open(os.path.join(bad_chunk_dir, f"bad_chunk_{chunk_id}.txt"), "w", encoding="utf-8") as f:
            f.write("=== CHUNK TEXT ===\n")
            f.write(chunk_text)
            f.write("\n\n=== RAW LLM RESPONSE ===\n")
            f.write(raw_text)

    def process_single_chunk(self, chunk_data: Dict) -> List[Dict]:
        """Worker 函數"""
        chunk_id = chunk_data['id']
        text = chunk_data['text']
        
        full_prompt = f"{self.system_prompt}\n\n【輸入文本】:\n{text}\n\n【輸出 JSON List】:"
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            # 先拿掉 format="json"（容易變慢/卡住）
            # "format": "json",
            "keep_alive": "30m",  # 讓模型別每次卸載
            "options": {
                "temperature": 0.0, 
                "num_ctx": 4096,      # 你的 prompt 很長，1024 很容易擠爆
                "num_predict": 1024,   # 關鍵：限制輸出，避免卡死 (Increased from 256 to 1024)
            }
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=900) 
            if response.status_code == 200:
                raw_text = response.json().get('response', '')
                extracted = self._clean_json_response(raw_text)
                
                if not extracted: # If no data extracted, dump the raw response for debugging
                    self._dump_bad_response(chunk_id, text, raw_text)

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

    def make_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        chunks = []
        start = 0
        chunk_counter = 0
        text = text.replace("\n", "") 

        # Ensure Chunk directory exists (created in run method)
        chunk_dir = os.path.join(os.path.dirname(OUTPUT_FILE), "Chunks")

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunk_id = chunk_counter
            chunks.append({
                "id": chunk_id,
                "text": chunk_text
            })
            
            # Save raw chunk to file
            with open(os.path.join(chunk_dir, f"chunk_{chunk_id}.txt"), "w", encoding="utf-8") as f:
                f.write(chunk_text)

            chunk_counter += 1
            start += (chunk_size - overlap)
            
        logging.info(f"文本已切分為 {len(chunks)} 個區塊 (Size: {chunk_size}, Overlap: {overlap})")
        return chunks

    def run(self, input_path: str, output_path: str, debug_mode: bool):
        self._check_ollama() # Check Ollama server status first
        raw_text = self.read_pdf(input_path, debug_mode)
        if not raw_text: return
        
        # Create output directories
        output_dir = os.path.dirname(output_path)
        chunk_dir = os.path.join(output_dir, "Chunks")
        bad_chunk_dir = os.path.join(output_dir, "Bad_Chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        os.makedirs(bad_chunk_dir, exist_ok=True)
        
        logging.info(f"結果將即時寫入: {output_path}")
        logging.info(f"原始區塊將儲存至: {chunk_dir}")
        logging.info(f"錯誤回應將儲存至: {bad_chunk_dir}")

        chunks = self.make_chunks(raw_text)
        
        logging.info(f"開始並行處理 (Workers: {MAX_WORKERS})...")

        # 如果是 Append 模式，這裡可以改成 'a'，但為了測試方便先用 'w'
        with open(output_path, 'w', encoding='utf-8') as f:
            pass

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_chunk = {executor.submit(self.process_single_chunk, chunk): chunk for chunk in chunks}
            
            total_entities = 0

            for future in tqdm(as_completed(future_to_chunk), total=len(chunks), desc="Processing Chunks"):
                result_data = future.result()
                
                if result_data:
                    total_entities += len(result_data)
                    with self.lock:
                        with open(output_path, 'a', encoding='utf-8') as f:
                            for item in result_data:
                                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    
                    first_person = result_data[0].get('name', 'Unknown')
                    # logging.info(f"提取 {len(result_data)} 筆 (如: {first_person})") # Tqdm handles progress, no need for this verbose logging
                # else:
                    # No need for explicit "無資料或處理完畢" as tqdm shows overall progress

        logging.info(f"任務完成！共提取 {total_entities} 個實體。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Debug mode: Process only first 10 pages")
    args = parser.parse_args()

    print("模式：提取所有實體 (Full Extraction)")

    pipeline = MingShiluNERPipeline(MODEL_NAME, API_URL) 
    pipeline.run(INPUT_FILE, OUTPUT_FILE, args.debug)
