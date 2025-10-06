#!/usr/bin/env python3
"""Verify all assignment requirements are met"""

import os
from pathlib import Path
import torch
import json

def verify_submission():
    print("\n" + "="*50)
    print("ASSIGNMENT 1 SUBMISSION VERIFICATION")
    print("="*50 + "\n")
    
    requirements = {
        "data_collection_preprocessing.py": False,
        "sample_dataset.pt": False,
        "Assignment1_Report.pdf": False,
        "1GB+ dataset": False,
        "Multiple domains": False,
        "Preprocessing complete": False,
        "Tokenization complete": False,
        "DataLoader implemented": False
    }
    
    # Check main script
    if Path("data_collection_preprocessing.py").exists():
        requirements["data_collection_preprocessing.py"] = True
        print("✅ data_collection_preprocessing.py found")
    else:
        print("❌ data_collection_preprocessing.py missing")
    
    # Check sample dataset
    if Path("outputs/sample_dataset.pt").exists():
        sample = torch.load("outputs/sample_dataset.pt")
        requirements["sample_dataset.pt"] = True
        print(f"✅ sample_dataset.pt found ({len(sample)} samples)")
    else:
        print("❌ sample_dataset.pt missing")
    
    # Check report
    if Path("Assignment1_Report.pdf").exists():
        requirements["Assignment1_Report.pdf"] = True
        print("✅ Assignment1_Report.pdf found")
    else:
        print("❌ Assignment1_Report.pdf missing")
    
    # Check dataset size
    total_size = 0
    for domain in ['encyclopedic', 'news', 'web']:
        path = Path(f"data/raw/{domain}_raw.json")
        if path.exists():
            total_size += path.stat().st_size
            print(f"✅ {domain} data found: {path.stat().st_size / (1024*1024):.1f}MB")
    
    if total_size >= 1024 * 1024 * 1024:  # 1GB
        requirements["1GB+ dataset"] = True
        print(f"✅ Total dataset size: {total_size / (1024*1024*1024):.2f}GB")
    else:
        print(f"❌ Dataset too small: {total_size / (1024*1024*1024):.2f}GB")
    
    # Check domains
    domains_found = []
    for domain in ['encyclopedic', 'news', 'web']:
        if Path(f"data/raw/{domain}_raw.json").exists():
            domains_found.append(domain)
    
    if len(domains_found) >= 2:
        requirements["Multiple domains"] = True
        print(f"✅ Multiple domains found: {', '.join(domains_found)}")
    
    # Check preprocessing
    if Path("data/processed/cleaned_texts.json").exists():
        requirements["Preprocessing complete"] = True
        print("✅ Preprocessing complete")
    
    # Check tokenization
    if Path("data/tokenized/tokenized_chunks.pt").exists():
        requirements["Tokenization complete"] = True
        print("✅ Tokenization complete")
    
    # Summary
    print("\n" + "="*50)
    passed = sum(requirements.values())
    total = len(requirements)
    
    if passed == total:
        print(f"🎉 ALL REQUIREMENTS MET! ({passed}/{total})")
        print("Ready for submission!")
    else:
        print(f"⚠️  {passed}/{total} requirements met")
        print("Missing:")
        for req, met in requirements.items():
            if not met:
                print(f"  - {req}")
    
    print("="*50)

if __name__ == "__main__":
    verify_submission()