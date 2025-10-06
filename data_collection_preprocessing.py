#!/usr/bin/env python3
"""
Assignment 1: Data Collection and Preprocessing for Foundation Model Pre-Training
Author: Ashley Han
Date: 2025
"""

import os
import json
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Tuple
import logging

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# PART 1: DATA COLLECTION
# ============================================================================

class DataCollector:
    """Handles downloading and collecting 1GB+ of diverse text data"""
    
    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_datasets(self) -> Dict[str, List[str]]:
        """
        Download 1GB+ of text from multiple domains:
        - bookcorpus (encyclopedic)
        - CC-News or similar (news)
        - OpenWebText or C4 (general web)
        """
        logger.info("Starting data collection for 1GB+ dataset")
        
        datasets = {
            'encyclopedic': [],
            'news': [],
            'web': []
        }
        
        target_size = 1.1 * 1024 * 1024 * 1024  # 1.1GB to ensure we exceed 1GB
        current_size = 0
        
        # 1. bookcorpus (Encyclopedic) - ~400MB
        logger.info("Downloading BookCorpus (encyclopedic content)...")
        try:
            wiki = load_dataset(
                "bookcorpus", 
                split="train",
                streaming=True
            )
            
            wiki_target = 700 * 1024 * 1024
            wiki_size = 0
            
            for item in tqdm(wiki, desc="bookcorpus", unit=" docs"):
                if wiki_size >= wiki_target:
                    break
                text = item['text']
                if len(text.split()) >= 50:  # Filter very short articles
                    datasets['encyclopedic'].append(text)
                    text_bytes = len(text.encode('utf-8'))
                    wiki_size += text_bytes
                    current_size += text_bytes
                    
            logger.info(f"✓ bookcorpus: {wiki_size/(1024*1024):.1f}MB collected")
            
        except Exception as e:
            logger.error(f"bookcorpus download failed: {e}")
            
        # 2. News Content - ~350MB
        logger.info("Downloading news content...")
        try:
            # Try CC-News first
            news = load_dataset("cc_news", split="train", streaming=True)
            news_target = 550 * 1024 * 1024
            news_size = 0
            
            for item in tqdm(news, desc="CC-News", unit=" articles"):
                if news_size >= news_target:
                    break
                text = item.get('text', '')
                if len(text.split()) >= 100:
                    datasets['news'].append(text)
                    text_bytes = len(text.encode('utf-8'))
                    news_size += text_bytes
                    current_size += text_bytes
                    
            logger.info(f"✓ News: {news_size/(1024*1024):.1f}MB collected")
            
        except Exception as e:
            logger.warning(f"CC-News failed, trying alternative: {e}")
            # Fallback to multi_news
            try:
                news = load_dataset("multi_news", split="train[:10000]")
                for item in news:
                    text = item.get('document', '')
                    datasets['news'].append(text)
                    current_size += len(text.encode('utf-8'))
            except Exception as e2:
                logger.error(f"News download failed: {e2}")
        
        # 3. General Web Text - ~350MB
        logger.info("Downloading general web text...")
        try:
            web = load_dataset(
                "wikitext",
                "wikitext-103-v1",
                split="train",
                streaming=True
            )
            
            web_target = 550 * 1024 * 1024
            web_size = 0
            
            for item in tqdm(web, desc="OpenWebText", unit=" docs"):
                if web_size >= web_target:
                    break
                text = item['text']
                if len(text.split()) >= 50:
                    datasets['web'].append(text)
                    text_bytes = len(text.encode('utf-8'))
                    web_size += text_bytes
                    current_size += text_bytes
                    
            logger.info(f"✓ Web text: {web_size/(1024*1024):.1f}MB collected")
            
        except Exception as e:
            logger.warning(f"OpenWebText failed, trying C4: {e}")
            # Fallback to C4
            try:
                c4 = load_dataset("c4", "en", split="train", streaming=True)
                web_target = 350 * 1024 * 1024
                web_size = 0
                
                for item in tqdm(c4, desc="C4", unit=" docs"):
                    if web_size >= web_target:
                        break
                    text = item['text']
                    datasets['web'].append(text)
                    text_bytes = len(text.encode('utf-8'))
                    web_size += text_bytes
                    current_size += text_bytes
                    
            except Exception as e2:
                logger.error(f"Web text download failed: {e2}")
        
        # Save raw datasets
        logger.info("Saving raw datasets...")
        for domain, texts in datasets.items():
            output_path = self.output_dir / f"{domain}_raw.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(texts, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(texts)} {domain} documents to {output_path}")
        
        # Print statistics
        total_docs = sum(len(texts) for texts in datasets.values())
        total_gb = current_size / (1024 * 1024 * 1024)
        
        logger.info(f"""
        ========================================
        DATA COLLECTION COMPLETE
        ========================================
        Total size: {total_gb:.2f} GB
        Total documents: {total_docs:,}
        
        Breakdown by domain:
        - Encyclopedic: {len(datasets['encyclopedic']):,} docs
        - News: {len(datasets['news']):,} docs
        - Web: {len(datasets['web']):,} docs
        ========================================
        """)
        
        return datasets

# ============================================================================
# PART 2: DATA CLEANING AND PREPROCESSING
# ============================================================================

class TextPreprocessor:
    """Handles all text cleaning and preprocessing operations"""
    
    def __init__(self):
        self.seen_hashes = set()
        self.stats = {
            'total_docs': 0,
            'removed_duplicates': 0,
            'removed_short': 0,
            'removed_low_quality': 0,
            'cleaned_docs': 0
        }
    
    def clean_html(self, text: str) -> str:
        """Remove HTML tags and decode entities"""
        if '<' in text and '>' in text:
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text()
        return text
    
    def normalize_text(self, text: str) -> str:
        """Normalize text: whitespace, special chars, etc."""
        # Remove HTML
        text = self.clean_html(text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove reference markers [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove markdown formatting
        text = re.sub(r'[#*`_~]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Normalize quotes and punctuation
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('—', '-').replace('–', '-')
        
        # Remove excessive punctuation
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        return text.strip()
    
    def is_quality_text(self, text: str) -> bool:
        """Check if text meets quality standards"""
        # Check minimum length
        words = text.split()
        if len(words) < 50:
            self.stats['removed_short'] += 1
            return False
        
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?;:\'-]', text)) / max(len(text), 1)
        if special_char_ratio > 0.3:
            self.stats['removed_low_quality'] += 1
            return False
        
        # Check for repetitive content
        unique_words = len(set(words))
        if unique_words < len(words) * 0.3:  # Too repetitive
            self.stats['removed_low_quality'] += 1
            return False
        
        return True
    
    def deduplicate(self, text: str) -> bool:
        """Check for duplicates using hash"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.seen_hashes:
            self.stats['removed_duplicates'] += 1
            return False
        self.seen_hashes.add(text_hash)
        return True
    
    def preprocess_dataset(self, datasets: Dict[str, List[str]]) -> List[str]:
        """Apply all preprocessing steps to the dataset"""
        logger.info("Starting preprocessing pipeline...")
        
        processed_texts = []
        
        for domain, texts in datasets.items():
            logger.info(f"Processing {domain} documents...")
            
            for text in tqdm(texts, desc=f"Processing {domain}"):
                self.stats['total_docs'] += 1
                
                # 1. Normalize text
                text = self.normalize_text(text)
                
                # 2. Quality check
                if not self.is_quality_text(text):
                    continue
                
                # 3. Deduplicate
                if not self.deduplicate(text):
                    continue
                
                processed_texts.append(text)
                self.stats['cleaned_docs'] += 1
        
        # Save processed data
        output_path = Path("data/processed/cleaned_texts.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_texts, f, ensure_ascii=False, indent=2)
        
        logger.info(f"""
        ========================================
        PREPROCESSING COMPLETE
        ========================================
        Total documents: {self.stats['total_docs']}
        Removed duplicates: {self.stats['removed_duplicates']}
        Removed short: {self.stats['removed_short']}
        Removed low quality: {self.stats['removed_low_quality']}
        Final cleaned documents: {self.stats['cleaned_docs']}
        
        Retention rate: {self.stats['cleaned_docs']/max(self.stats['total_docs'],1)*100:.1f}%
        ========================================
        """)
        
        return processed_texts

# ============================================================================
# PART 3: TOKENIZATION
# ============================================================================

class TokenizationPipeline:
    """Handles tokenization for transformer models"""
    
    def __init__(self, model_name="gpt2", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.vocab_size = len(self.tokenizer)
        
        logger.info(f"""
        Tokenizer initialized:
        - Model: {model_name}
        - Vocabulary size: {self.vocab_size}
        - Max sequence length: {max_length}
        """)
    
    def tokenize_and_chunk(self, texts: List[str]) -> List[torch.Tensor]:
        """Tokenize texts and chunk into fixed-length sequences"""
        logger.info("Starting tokenization...")
        
        all_chunks = []
        
        for text in tqdm(texts, desc="Tokenizing"):
            # Tokenize without truncation
            tokens = self.tokenizer(
                text,
                truncation=False,
                return_tensors="pt"
            )
            
            input_ids = tokens["input_ids"][0]
            
            # Chunk into max_length sequences with stride
            stride = self.max_length // 2  # 50% overlap
            
            for i in range(0, len(input_ids) - self.max_length + 1, stride):
                chunk = input_ids[i:i + self.max_length]
                if len(chunk) == self.max_length:
                    all_chunks.append(chunk)
        
        logger.info(f"Created {len(all_chunks)} chunks of size {self.max_length}")
        
        # Save tokenized data
        output_path = Path("data/tokenized/tokenized_chunks.pt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(all_chunks, output_path)
        
        # Save sample for submission
        sample_path = Path("outputs/sample_dataset.pt")
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(all_chunks[:10], sample_path)  # First 10 chunks
        
        return all_chunks

# ============================================================================
# PART 4: CUSTOM DATA LOADER
# ============================================================================

class PretrainingDataset(Dataset):
    """Custom PyTorch Dataset for pretraining"""
    
    def __init__(self, tokenized_chunks):
        self.data = tokenized_chunks
        logger.info(f"Dataset created with {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # For causal LM, labels are the same as input_ids
        return {
            'input_ids': self.data[idx],
            'labels': self.data[idx].clone(),
            'attention_mask': torch.ones_like(self.data[idx])
        }

def create_data_loader(tokenized_chunks, batch_size=8, shuffle=True):
    """Create PyTorch DataLoader with optimizations"""
    
    dataset = PretrainingDataset(tokenized_chunks)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster GPU transfer
        prefetch_factor=2,  # Prefetch batches
        persistent_workers=True  # Keep workers alive
    )
    
    logger.info(f"""
    DataLoader created:
    - Batch size: {batch_size}
    - Shuffle: {shuffle}
    - Num workers: 4
    - Total batches: {len(dataloader)}
    """)
    
    return dataloader

# ============================================================================
# PART 5: DATA QUALITY ANALYSIS (Optional Extension)
# ============================================================================

def analyze_data_quality(tokenized_chunks, tokenizer):
    """Analyze tokenized data quality"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    logger.info("Analyzing data quality...")
    
    # Token length distribution
    lengths = [len(chunk) for chunk in tokenized_chunks]
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Sequence length distribution
    axes[0, 0].hist(lengths, bins=50, edgecolor='black')
    axes[0, 0].set_title('Sequence Length Distribution')
    axes[0, 0].set_xlabel('Length')
    axes[0, 0].set_ylabel('Count')
    
    # 2. Token frequency analysis
    all_tokens = torch.cat(tokenized_chunks).numpy()
    unique_tokens, counts = np.unique(all_tokens, return_counts=True)
    
    top_20_indices = np.argsort(counts)[-20:]
    top_20_tokens = unique_tokens[top_20_indices]
    top_20_counts = counts[top_20_indices]
    
    axes[0, 1].barh(range(20), top_20_counts)
    axes[0, 1].set_title('Top 20 Most Frequent Tokens')
    axes[0, 1].set_xlabel('Frequency')
    
    # 3. Vocabulary coverage
    vocab_coverage = len(unique_tokens) / len(tokenizer) * 100
    axes[1, 0].text(0.5, 0.5, f'Vocabulary Coverage:\n{vocab_coverage:.1f}%\n\n'
                    f'Unique tokens used: {len(unique_tokens):,}\n'
                    f'Total vocabulary: {len(tokenizer):,}',
                    ha='center', va='center', fontsize=12)
    axes[1, 0].axis('off')
    
    # 4. Statistics summary
    stats_text = f"""
    Dataset Statistics:
    - Total sequences: {len(tokenized_chunks):,}
    - Total tokens: {len(all_tokens):,}
    - Avg sequence length: {np.mean(lengths):.1f}
    - Min sequence length: {np.min(lengths)}
    - Max sequence length: {np.max(lengths)}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, ha='left', va='center', fontsize=10)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/data_quality_analysis.png', dpi=300)
    plt.show()
    
    logger.info(f"Vocabulary coverage: {vocab_coverage:.1f}%")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    
    logger.info("""
    ================================================
    ASSIGNMENT 1: DATA COLLECTION AND PREPROCESSING
    ================================================
    """)
    
    # Step 1: Data Collection
    logger.info("\n[STEP 1/5] DATA COLLECTION")
    collector = DataCollector()
    datasets = collector.collect_datasets()
    
    # Step 2: Preprocessing
    logger.info("\n[STEP 2/5] PREPROCESSING")
    preprocessor = TextPreprocessor()
    cleaned_texts = preprocessor.preprocess_dataset(datasets)
    
    # Step 3: Tokenization
    logger.info("\n[STEP 3/5] TOKENIZATION")
    tokenizer_pipeline = TokenizationPipeline(model_name="gpt2", max_length=512)
    tokenized_chunks = tokenizer_pipeline.tokenize_and_chunk(cleaned_texts)
    
    # Step 4: Create Data Loader
    logger.info("\n[STEP 4/5] DATA LOADER CREATION")
    dataloader = create_data_loader(tokenized_chunks, batch_size=8)
    
    # Test the dataloader
    logger.info("Testing dataloader...")
    for i, batch in enumerate(dataloader):
        if i >= 2:  # Test first 2 batches
            break
        logger.info(f"Batch {i+1} shape: {batch['input_ids'].shape}")
    
    # Step 5: Data Quality Analysis
    logger.info("\n[STEP 5/5] DATA QUALITY ANALYSIS")
    analyze_data_quality(tokenized_chunks, tokenizer_pipeline.tokenizer)
    
    logger.info("""
    ================================================
    PIPELINE COMPLETE!
    
    Deliverables created:
    1. ✓ data_collection_preprocessing.py (this file)
    2. ✓ sample_dataset.pt (in outputs/)
    3. ✓ data_quality_analysis.png (in outputs/)
    
    Next: Create Assignment1_Report.pdf
    ================================================
    """)

if __name__ == "__main__":
    # Create necessary directories
    for dir_name in ['logs', 'outputs']:
        Path(dir_name).mkdir(exist_ok=True)
    
    # Run the pipeline
    main()