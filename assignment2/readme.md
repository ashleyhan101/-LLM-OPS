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
