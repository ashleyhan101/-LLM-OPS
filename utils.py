"""
Utility functions for checkpointing, logging, and visualization
"""
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'loss': loss,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    torch.save(checkpoint, filepath)
    print(f"✓ Checkpoint saved: {filepath}")


def plot_training_curves(losses, perplexities, save_path='outputs/plots/training_curves.png'):
    """Plot training loss and perplexity"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(losses) + 1)
    
    ax1.plot(epochs, losses, 'b-', linewidth=2, marker='o')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, perplexities, 'r-', linewidth=2, marker='o')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Perplexity', fontsize=12)
    ax2.set_title('Training Perplexity', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved: {save_path}")


def save_metrics(losses, perplexities, save_path='outputs/logs/metrics.txt'):
    """Save training metrics"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("TRAINING METRICS\n")
        f.write("=" * 50 + "\n\n")
        for epoch, (loss, perp) in enumerate(zip(losses, perplexities), 1):
            f.write(f"Epoch {epoch}: Loss={loss:.4f}, Perplexity={perp:.2f}\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Final Loss: {losses[-1]:.4f}\n")
        f.write(f"Final Perplexity: {perplexities[-1]:.2f}\n")
        f.write(f"Best Loss: {min(losses):.4f}\n")
        f.write(f"Best Perplexity: {min(perplexities):.2f}\n")
    
    print(f"✓ Metrics saved: {save_path}")


def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and checkpoint['optimizer_state_dict']:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"✓ Checkpoint loaded from epoch {epoch}, loss: {loss:.4f}")
    return model, optimizer, epoch, loss