#!/bin/bash

# Check if timestamp parameter is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide the model timestamp as parameter"
    echo "Usage: ./run_evaluation.sh TIMESTAMP"
    echo "Example: ./run_evaluation.sh 20250201_003628"
    exit 1
fi

TIMESTAMP=$1
echo "Running evaluation for timestamp: $TIMESTAMP"

# Create necessary directories
echo "Creating directories..."
mkdir -p data/scorers/subtask1_baselines/EN
mkdir -p data/scorers/subtask1_baselines/PT

# Function to check command success
check_status() {
    if [ $? -eq 0 ]; then
        echo "✓ Success: $1"
    else
        echo "✗ Failed: $1"
        exit 1
    fi
}

# Function to extract metrics from scorer output
extract_metrics() {
    # Read from stdin and extract metrics
    while IFS= read -r line; do
        if [[ $line =~ ^[0-9]+\.[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+[0-9]+\.[0-9]+[[:space:]]+[0-9]+\.[0-9]+$ ]]; then
            # Split the line into array
            read -ra METRICS <<< "$line"
            # EMR, Micro_P, Micro_R, Micro_F1, Main_Role_Acc
            echo "${METRICS[0]},${METRICS[1]},${METRICS[2]},${METRICS[3]},${METRICS[4]}"
        fi
    done
}

# Function to format metrics
format_metrics() {
    IFS=',' read emr precision recall f1 main_acc <<< "$1"
    echo "  EMR=${emr}"
    echo "  Precision=${precision}"
    echo "  Recall=${recall}"
    echo "  F1=${f1}"
    echo "  Main Role Accuracy=${main_acc}"
    echo ""
}

echo -e "\n=== Generating Baselines ==="

# Generate English baselines
echo -e "\nGenerating English baselines..."
python data/scorers/subtask1_baseline.py \
    --dev_file data/dev/EN/subtask-1-entity-mentions.txt \
    --output_dir data/scorers/subtask1_baselines/EN \
    --baseline_type random
check_status "English random baseline"

python data/scorers/subtask1_baseline.py \
    --train_file data/train/EN/subtask-1-annotations.txt \
    --dev_file data/dev/EN/subtask-1-entity-mentions.txt \
    --output_dir data/scorers/subtask1_baselines/EN \
    --baseline_type majority
check_status "English majority baseline"

# Generate Portuguese baselines
echo -e "\nGenerating Portuguese baselines..."
python data/scorers/subtask1_baseline.py \
    --dev_file data/dev/PT/subtask-1-entity-mentions.txt \
    --output_dir data/scorers/subtask1_baselines/PT \
    --baseline_type random
check_status "Portuguese random baseline"

python data/scorers/subtask1_baseline.py \
    --train_file data/train/PT/subtask-1-annotations.txt \
    --dev_file data/dev/PT/subtask-1-entity-mentions.txt \
    --output_dir data/scorers/subtask1_baselines/PT \
    --baseline_type majority
check_status "Portuguese majority baseline"

echo -e "\n=== Running Scorers ==="

# Score and collect metrics for English
echo -e "\n### English (EN)"

# Model predictions
echo "- Model Performance:"
METRICS=$(python data/scorers/subtask1_scorer.py \
    --gold_file data/dev/EN/subtask-1-annotations.txt \
    --pred_file predictions/predictions_EN_eval_${TIMESTAMP}.txt | extract_metrics)
format_metrics "$METRICS"
check_status "English predictions scoring"

# Random baseline
echo "- Random Baseline:"
METRICS=$(python data/scorers/subtask1_scorer.py \
    --gold_file data/dev/EN/subtask-1-annotations.txt \
    --pred_file data/scorers/subtask1_baselines/EN/baseline_random.txt | extract_metrics)
format_metrics "$METRICS"
check_status "English random baseline scoring"

# Majority baseline
echo "- Majority Baseline:"
METRICS=$(python data/scorers/subtask1_scorer.py \
    --gold_file data/dev/EN/subtask-1-annotations.txt \
    --pred_file data/scorers/subtask1_baselines/EN/baseline_majority.txt | extract_metrics)
format_metrics "$METRICS"
check_status "English majority baseline scoring"

# Score and collect metrics for Portuguese
echo -e "\n### Portuguese (PT)"

# Model predictions
echo "- Model Performance:"
METRICS=$(python data/scorers/subtask1_scorer.py \
    --gold_file data/dev/PT/subtask-1-annotations.txt \
    --pred_file predictions/predictions_PT_eval_${TIMESTAMP}.txt | extract_metrics)
format_metrics "$METRICS"
check_status "Portuguese predictions scoring"

# Random baseline
echo "- Random Baseline:"
METRICS=$(python data/scorers/subtask1_scorer.py \
    --gold_file data/dev/PT/subtask-1-annotations.txt \
    --pred_file data/scorers/subtask1_baselines/PT/baseline_random.txt | extract_metrics)
format_metrics "$METRICS"
check_status "Portuguese random baseline scoring"

# Majority baseline
echo "- Majority Baseline:"
METRICS=$(python data/scorers/subtask1_scorer.py \
    --gold_file data/dev/PT/subtask-1-annotations.txt \
    --pred_file data/scorers/subtask1_baselines/PT/baseline_majority.txt | extract_metrics)
format_metrics "$METRICS"
check_status "Portuguese majority baseline scoring"

echo -e "\n=== Evaluation Complete ===" 