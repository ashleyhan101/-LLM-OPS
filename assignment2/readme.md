# Mini-GPT: Small-Scale Foundation Model

Implementation of a small-scale transformer-based language model for next-token prediction.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your data:
   - Place tokenized data in `data/processed/tokenized_data.txt`

3. Configure hyperparameters in `config.py`

## Usage

### Training
```bash
python main.py
```

### Hyperparameter Experiments
```bash
python hyperparameter_search.py
```

## Project Structure
- `src/`: Core implementation
- `checkpoints/`: Saved models
- `outputs/`: Training curves and logs
- `notebooks/`: Jupyter notebooks for analysis

## Results
- Training curves: `outputs/plots/training_curves.png`
- Model checkpoints: `checkpoints/mini_gpt_final.pt`

## Model Checkpoint

Due to the large file size (70MB), the checkpoint is not uploaded to the repository.

### Reproducing Results

To generate the checkpoint, run:
```bash
python3 main.py --config config.py
```

This will automatically train the model and save the checkpoint to `checkpoints/best_model.pt`

### Project Files

- `model.py` - Mini-GPT Model Implementation (Multi-head attention, Transformer blocks)
- `train.py` - Training loop, checkpoint saving/loading, loss and perplexity logging
- `src/` - Data processing and utility functions
- `report-Yazhen.pdf` - Complete project report
