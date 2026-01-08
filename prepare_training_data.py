import pandas as pd
import ast
import re
import json
import logging

# --- Configuration ---
INPUT_CSV = "official%2525252525252525252525252525252525252525252525252525252525252525252525252525252525252.csv"
OUTPUT_JSONL = "train.jsonl"

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_ming_official_row(text_content):
    """
    Parses the raw text from MingOfficial which contains tags like <P>...</P> and <O>...</O>.
    Returns:
        clean_text: Text with all tags removed.
        entities: List of dicts {'name': ..., 'rank': ...} extracted from tags.
    """
    entities = []
    
    # Find all <O>...</O> and <P>...</P> tags in order
    # Regex captures: Group 2 is Rank text, Group 4 is Person Name
    tokens = re.finditer(r'(<O>(.*?)</O>)|(<P.*?>(.*?)</P>)', text_content)
    
    last_rank = None
    last_rank_end_pos = -1
    
    for match in tokens:
        if match.group(1):  # It's an <O> tag (Rank)
            last_rank = match.group(2)
            last_rank_end_pos = match.end()
        elif match.group(3):  # It's a <P> tag (Person)
            name = match.group(4)
            current_start_pos = match.start()
            
            # Check if this Person immediately follows the last Rank
            # (Allows a small gap of < 5 chars for spaces or connecting words)
            rank = None
            if last_rank and (current_start_pos - last_rank_end_pos) < 5: 
                rank = last_rank
                last_rank = None  # Consume the rank
            
            entities.append({
                "name": name,
                "rank": rank,
                "event_type": "N/A",  # MingOfficial tags don't have event info
                "action": "N/A"       # MingOfficial tags don't have action info
            })
            
    # Clean text: remove all tags <...>
    clean_text = re.sub(r'<[^>]+>', '', text_content)
    
    return clean_text, entities

def main():
    training_data = []
    
    try:
        logging.info(f"Reading CSV file: {INPUT_CSV}")
        # Read CSV in chunks to handle large files
        chunk_size = 1000
        chunks_processed = 0
        
        for chunk in pd.read_csv(INPUT_CSV, chunksize=chunk_size):
            chunks_processed += 1
            logging.info(f"Processing chunk {chunks_processed} ({len(chunk)} rows)...")
            
            for index, row in chunk.iterrows():
                related_texts_str = row['Related Texts']
                
                if pd.isna(related_texts_str) or related_texts_str == "N/A":
                    continue
                    
                try:
                    # Parse the stringified list of tuples
                    related_texts = ast.literal_eval(related_texts_str)
                    
                    for item in related_texts:
                        # item structure: (id, source, text_content)
                        if len(item) >= 3:
                            text_content = item[2]
                            clean_input, ground_truth_entities = parse_ming_official_row(text_content)
                            
                            # Only add if we extracted at least one entity
                            if ground_truth_entities:
                                # Construct the training example
                                example = {
                                    "instruction": "你是明朝歷史資料專家。請從以下文本中提取官員資訊，輸出為 JSON List。",
                                    "input": clean_input,
                                    "output": json.dumps(ground_truth_entities, ensure_ascii=False)
                                }
                                training_data.append(example)
                        
                except Exception as e:
                    # Skip malformed rows
                    continue
        
        # Save to JSONL
        logging.info(f"Saving {len(training_data)} training examples to {OUTPUT_JSONL}")
        with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
            for entry in training_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logging.info(f"Done! Generated {len(training_data)} training examples in {OUTPUT_JSONL}")
        
        # Print first 2 examples for verification
        if training_data:
            logging.info("\n=== Sample Training Examples ===")
            for i in range(min(2, len(training_data))):
                logging.info(f"\nExample {i+1}:")
                logging.info(f"Instruction: {training_data[i]['instruction']}")
                logging.info(f"Input: {training_data[i]['input'][:100]}...")
                logging.info(f"Output: {training_data[i]['output'][:200]}...")
                
    except FileNotFoundError:
        logging.error(f"Error: {INPUT_CSV} not found. Please provide the correct path.")
        return
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        raise

if __name__ == "__main__":
    main()
