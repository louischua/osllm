r"""
Download and prepare training data from the SQUAD dataset.

OVERVIEW:
This script downloads the SQUAD (Stanford Question Answering Dataset) from its official source,
extracts the Wikipedia context passages from the JSON format, and saves the cleaned text to disk.
The SQUAD dataset contains high-quality Wikipedia articles that are perfect for training language models.

DATA FLOW:
1. Downloads 4 JSON files from Stanford (SQUAD v1.1 & v2.0, train & dev splits)
2. Parses JSON structure: data -> articles -> paragraphs -> context
3. Extracts only the 'context' fields (Wikipedia passages, not questions/answers)  
4. Cleans text: normalizes whitespace, filters by minimum word count
5. Outputs one passage per line in a single text file

The output is a single text file containing ~150k-200k Wikipedia article passages, 
suitable for training tokenizers and language models.

DATASET INFO:
- SQUAD v1.1: 87k train + 10k dev examples
- SQUAD v2.0: 130k train + 11k dev examples  
- Source: High-quality Wikipedia articles across diverse topics
- Total download size: ~200MB
- Final processed size: ~100-150MB of clean text

Usage:
    python core/src/download_and_prepare.py

Output:
    data/clean/training_data.txt - Cleaned Wikipedia passages from SQUAD dataset

Requirements:
    pip install requests tqdm

Example setup:

Windows PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install requests tqdm
python core/src/download_and_prepare.py
```

Linux/macOS:
```bash
python -m venv venv
source venv/bin/activate
pip install requests tqdm
python core/src/download_and_prepare.py
```

"""

import os
import requests
import json
from tqdm import tqdm

def download_file(url, filename):
    """
    Download a file from URL with progress bar.
    
    Args:
        url (str): URL to download from
        filename (str): Local path where file should be saved
    """
    # Stream the download to handle large files efficiently
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    # Use tqdm progress bar to show download progress
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        # Download in 1KB chunks
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def prepare_training_data(output_path="data/clean/training_data.txt", min_words=10):
    """
    Downloads the SQUAD dataset and extracts Wikipedia context passages for training.
    
    SQUAD Dataset Structure:
    - Each JSON file contains a 'data' array of Wikipedia articles
    - Each article has 'paragraphs' containing 'context' (Wikipedia text) and 'qas' (questions/answers)
    - We extract only the 'context' fields which contain high-quality Wikipedia passages
    
    Args:
        output_path (str): Path to save the cleaned text data.
        min_words (int): Minimum number of words required for a passage to be included.
    """
    print("Downloading SQUAD dataset...")
    
    # Official SQUAD dataset URLs from Stanford
    # Using both v1.1 and v2.0 for maximum training data
    # v1.1: ~87k training + 10k dev examples
    # v2.0: ~130k training + 11k dev examples (includes unanswerable questions)
    squad_urls = [
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json", 
        "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
    ]
    
    # Create directory structure for temporary files
    os.makedirs("data/raw", exist_ok=True)
    
    downloaded_files = []
    
    # Download each SQUAD dataset file
    print("Step 1: Downloading SQUAD JSON files...")
    for i, url in enumerate(squad_urls):
        filename = f"data/raw/squad_{i+1}.json"
        try:
            print(f"Downloading {url}...")
            download_file(url, filename)
            downloaded_files.append(filename)
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            continue
    
    # Verify we have at least one successful download
    if not downloaded_files:
        print("ERROR: No files were downloaded successfully.")
        return
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"\nStep 2: Processing SQUAD files and saving to {output_path}...")

    # Process each downloaded SQUAD JSON file and extract contexts
    with open(output_path, "w", encoding="utf-8") as f:
        total_contexts = 0
        
        for file_path in downloaded_files:
            print(f"Processing {file_path}...")
            
            try:
                # Load and parse the JSON file
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    squad_data = json.load(json_file)
                
                # Navigate the SQUAD JSON structure to extract context passages
                # Structure: data -> articles -> paragraphs -> context
                contexts = []
                for article in squad_data.get('data', []):
                    # Each article represents a Wikipedia page
                    for paragraph in article.get('paragraphs', []):
                        # Each paragraph contains a 'context' (Wikipedia passage) and 'qas' (Q&A pairs)
                        context = paragraph.get('context', '').strip()
                        if context:
                            contexts.append(context)
                
                print(f"Found {len(contexts)} Wikipedia passages in {os.path.basename(file_path)}")
                
                # Clean and filter each context passage for high-quality training data
                # This preprocessing is crucial for effective language model training
                for context in tqdm(contexts, desc=f"Processing {os.path.basename(file_path)}"):
                    
                    # Text normalization and cleaning pipeline
                    # Step 1: Normalize whitespace to ensure consistent formatting
                    # - Collapse multiple spaces/tabs into single spaces
                    # - Remove excessive newlines that break sentence flow
                    # - Strip leading/trailing whitespace
                    # This preserves natural sentence structure while cleaning artifacts
                    cleaned_text = ' '.join(context.split())
                    
                    # Step 2: Skip empty passages after cleaning
                    # Empty passages can occur from malformed JSON or pure whitespace
                    if not cleaned_text:
                        continue
                    
                    # Step 3: Quality filtering based on content length
                    # Apply minimum word count filter to ensure substantial content
                    # Short passages (< min_words) provide insufficient context for language modeling
                    # Wikipedia passages are typically well-formed, so this mainly catches truncated text
                    word_count = len(cleaned_text.split())
                    if word_count >= min_words:
                        # Write each passage on a new line for easy processing by data loaders
                        # The line-based format enables efficient streaming during training
                        # Each line represents one coherent Wikipedia passage
                        f.write(cleaned_text + '\n')
                        total_contexts += 1
                    
                    # Optional: Log extremely short passages for monitoring data quality
                    elif word_count > 0:  # Non-empty but too short
                        if total_contexts % 1000 == 0:  # Log occasionally to avoid spam
                            print(f"⚠️  Skipped short passage ({word_count} words): {cleaned_text[:50]}...")
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    print(f"\nStep 3: Successfully saved {total_contexts} Wikipedia passages from SQUAD dataset.")
    print(f"Output file: {output_path}")
    
    # Clean up temporary downloaded files to save disk space
    print("Step 4: Cleaning up temporary files...")
    for file in downloaded_files:
        try:
            os.remove(file)
            print(f"Removed {file}")
        except:
            pass

if __name__ == "__main__":
    """
    Main execution block - runs when script is called directly.
    
    This will:
    1. Download SQUAD v1.1 and v2.0 datasets (~200MB total)
    2. Extract ~240k Wikipedia passages from the JSON files  
    3. Clean and filter the text (remove passages < 10 words)
    4. Save all passages to data/clean/training_data.txt (one per line)
    5. Clean up temporary files
    
    Expected output: ~150k-200k high-quality Wikipedia passages suitable for LM training
    """
    # Run the data preparation function with default parameters
    prepare_training_data()
