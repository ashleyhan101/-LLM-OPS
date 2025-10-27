"""
Dataset and data loading utilities
"""
import torch
from torch.utils.data import Dataset, DataLoader


class TokenDataset(Dataset):
    def __init__(self, tokenized_chunks, seq_length=64):
        """
        Args:
            tokenized_chunks: List of tokenized sequences from Assignment 1
            seq_length: Length of each training sequence
        """
        # Flatten all chunks into one long sequence
        if isinstance(tokenized_chunks, list):
            if isinstance(tokenized_chunks[0], torch.Tensor):
                self.token_ids = torch.cat(tokenized_chunks).tolist()
            else:
                self.token_ids = tokenized_chunks
        else:
            self.token_ids = tokenized_chunks.tolist()
            
        self.seq_length = seq_length
        print(f"✓ Dataset initialized with {len(self.token_ids):,} tokens")
        
    def __len__(self):
        return max(0, len(self.token_ids) - self.seq_length)
    
    def __getitem__(self, idx):
        chunk = self.token_ids[idx:idx + self.seq_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_tokenized_data(filepath='data/processed/tokenized_chunks.pt'):
    """
    Load tokenized data from Assignment 1 (.pt format)
    
    Returns:
        Tuple of (tokenized_chunks, vocab_size)
    """
    print(f"Loading data from: {filepath}")
    
    # Load the PyTorch tensor file
    tokenized_chunks = torch.load(filepath)
    
    # Get vocabulary size from the data
    if isinstance(tokenized_chunks, list):
        all_tokens = torch.cat(tokenized_chunks)
    else:
        all_tokens = tokenized_chunks
    
    vocab_size = int(all_tokens.max().item()) + 1
    
    print(f"✓ Data loaded successfully")
    print(f"  - Number of chunks: {len(tokenized_chunks)}")
    print(f"  - Total tokens: {len(all_tokens):,}")
    print(f"  - Vocabulary size: {vocab_size:,}")
    
    return tokenized_chunks, vocab_size


def create_dataloader(tokenized_chunks, seq_length, batch_size, shuffle=True):
    """
    Create a DataLoader from tokenized chunks
    """
    dataset = TokenDataset(tokenized_chunks, seq_length)
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty! Check your tokenized data.")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"✓ DataLoader created: {len(dataset):,} sequences, {len(dataloader)} batches\n")
    return dataloader