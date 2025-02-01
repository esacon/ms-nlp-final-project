# Entity Framing Model: Training and Evaluation Guide

This guide provides step-by-step instructions for training, evaluating, and comparing the Entity Framing model.

## 1. Training the Model

To train the model on both English (EN) and Portuguese (PT) data:

```bash
python run.py --task train --languages EN PT --output_dir models
```

Options:
- `--max_articles N`: Limit training to N articles (useful for debugging)
- `--config path/to/config.yaml`: Use custom configuration file

## 2. Evaluating the Model

### 2.1 Run Model Evaluation

To evaluate the model on the development set:

```bash
python run.py \
  --task evaluate \
  --languages EN PT \
  --output_dir predictions \
  --checkpoint models/best_model_xlm-roberta-base_[TIMESTAMP].pt
```

This will:
- Load the specified model checkpoint
- Generate predictions for both languages
- Save predictions to `predictions/predictions_[LANG]_eval_[TIMESTAMP].txt`
- Run the scorer automatically for development set predictions

### 2.2 Generate Baselines

Generate random and majority baselines for comparison:

For English:
```bash
# Random baseline
python data/scorers/subtask1_baseline.py \
  --dev_file data/dev/EN/subtask-1-entity-mentions.txt \
  --output_dir data/scorers/subtask1_baselines/EN \
  --baseline_type random

# Majority baseline
python data/scorers/subtask1_baseline.py \
  --train_file data/train/EN/subtask-1-annotations.txt \
  --dev_file data/dev/EN/subtask-1-entity-mentions.txt \
  --output_dir data/scorers/subtask1_baselines/EN \
  --baseline_type majority
```

For Portuguese:
```bash
# Random baseline
python data/scorers/subtask1_baseline.py \
  --dev_file data/dev/PT/subtask-1-entity-mentions.txt \
  --output_dir data/scorers/subtask1_baselines/PT \
  --baseline_type random

# Majority baseline
python data/scorers/subtask1_baseline.py \
  --train_file data/train/PT/subtask-1-annotations.txt \
  --dev_file data/dev/PT/subtask-1-entity-mentions.txt \
  --output_dir data/scorers/subtask1_baselines/PT \
  --baseline_type majority
```

### 2.3 Run Scorers

To evaluate predictions against gold standard:

For English:
```bash
python data/scorers/subtask1_scorer.py \
  --gold_file data/dev/EN/subtask-1-annotations.txt \
  --pred_file predictions/predictions_EN_[TIMESTAMP].txt
```

For Portuguese:
```bash
python data/scorers/subtask1_scorer.py \
  --gold_file data/dev/PT/subtask-1-annotations.txt \
  --pred_file predictions/predictions_PT_[TIMESTAMP].txt
```

## 3. Understanding the Metrics

The scorer outputs the following metrics:
- **EMR (Exact Match Ratio)**: Percentage of perfectly predicted role hierarchies
- **Micro Precision**: Precision across all predictions
- **Micro Recall**: Recall across all predictions
- **Micro F1**: Harmonic mean of precision and recall
- **Main Role Accuracy**: Accuracy of main role classification only

## 4. Example Results

### English (EN)
- Model Performance: EMR=0.0110, F1=0.0314, Main Role Acc=0.7692
- Random Baseline: EMR=0.0220, F1=0.0314, Main Role Acc=0.2418
- Majority Baseline: EMR=0.0659, F1=0.0942, Main Role Acc=0.8022

### Portuguese (PT)
- Model Performance: EMR=0.0345, F1=0.0417, Main Role Acc=0.2845
- Random Baseline: EMR=0.0776, F1=0.0833, Main Role Acc=0.4138
- Majority Baseline: EMR=0.4483, F1=0.4667, Main Role Acc=0.2414

## 5. Troubleshooting

1. If evaluation fails with file size mismatch:
   - Ensure you're evaluating on the development set, not test set
   - Check that `--max_articles` parameter matches your intended evaluation scope

2. If model checkpoint isn't found:
   - Verify the checkpoint path exists
   - Use the most recent checkpoint from the models directory

3. For baseline generation:
   - Ensure all required data files exist in the specified paths
   - Check that output directories exist and are writable 