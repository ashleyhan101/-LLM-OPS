"""
Configuration file for Mini-GPT training
"""

class Config:
    # Model architecture
    VOCAB_SIZE = 50257  # GPT-2 vocab size (auto-detected)
    D_MODEL = 64        # Smaller model for sample data
    NUM_LAYERS = 1      # Single layer for testing
    NUM_HEADS = 2       # 2 attention heads
    D_FF = 256          # Smaller feed-forward
    MAX_SEQ_LEN = 64    # Sequence length
    DROPOUT = 0.3       # Higher dropout to reduce overfitting
    
    # Training parameters
    BATCH_SIZE = 4      # Small batch for sample
    LEARNING_RATE = 5e-4
    NUM_EPOCHS = 10     # Quick training for testing
    WEIGHT_DECAY = 0.1  # Higher weight decay
    GRAD_CLIP = 1.0
    
    # Data paths
    DATA_PATH = 'data/processed/tokenized_chunks.pt'
    CHECKPOINT_DIR = 'checkpoints/'
    OUTPUT_DIR = 'outputs/'
    
    # Device
    DEVICE = 'cpu'  # Change to 'cuda' if you have GPU
    
    # Logging
    LOG_INTERVAL = 5
    SAVE_INTERVAL = 5