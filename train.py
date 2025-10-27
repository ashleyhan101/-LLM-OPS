"""
Training loop and utilities
"""
import torch
import math
from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, device, grad_clip=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for x, y in progress_bar:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(x, targets=y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def calculate_perplexity(loss):
    """Calculate perplexity from loss"""
    try:
        return math.exp(loss)
    except OverflowError:
        return float('inf')