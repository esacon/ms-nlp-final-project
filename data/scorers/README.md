# Data for SemEval 2025 task 10

This Readme is distributed with the data for participating in SemEval 2025 task 10. 
The website of the shared task (https://propaganda.math.unipd.it/semeval2025task10/), includes a detailed descripton of the tasks, the submission instructions, updates on the competition and a live leaderboard.


## Task Description

### Subtask 1: Entity Framing

Given a news article and a list of mentions of named entities (NEs) in the article, assign for each such mention one or more roles using a predefined taxonomy of fine-grained roles covering three main type of roles: protagonists, antagonists, and innocent. This is a multi-label multi-class text-span classification task.

## Data Format 

The format of the input, gold label and prediction files is specified on the [website of the competition](https://propaganda.math.unipd.it/semeval2025task10/). Note that the scorer will only accept files with the .txt extension.


## Baselines and Scorers 

### Subtask 1

#### Baseline Code Usage

For subtask 1, we generate 2 baselines for each language: a random guessing baseline, and a majority voting baseline. To generate baselines for all four languages (BG, EN, HI, PT):

```bash

# For EN (English)
python subtask1_baseline.py --dev_file data/EN/subtask-1-entity-mentions.txt --output_dir subtask1_baselines/EN --baseline_type random
python subtask1_baseline.py --train_file data/EN/subtask-1-annotations.txt --dev_file data/EN/subtask-1-entity-mentions.txt --output_dir subtask1_baselines/EN --baseline_type majority

# For PT (Portuguese)
python subtask1_baseline.py --dev_file data/PT/subtask-1-entity-mentions.txt --output_dir subtask1_baselines/PT --baseline_type random
python subtask1_baseline.py --train_file data/PT/subtask-1-annotations.txt --dev_file data/PT/subtask-1-entity-mentions.txt --output_dir subtask1_baselines/PT --baseline_type majority

```

The predictions are saved as baseline_random.txt and baseline_majority.txt in the specified output directories under subtask1_baselines.

#### Scorer Code Usage
The subtask1_scorer.py script evaluates the baselines by comparing predictions against the gold labels. 


<!-- Use the following commands to score both the random and majority baselines for each language. The **-g** argument is for the ground truth file path and **-p** is for the predictions file path -->

```bash
python subtask1_scorer.py -g data/EN/subtask-1-entity-mentions.txt -p data/EN/subtask-1-entity-mentions.txt 
```
Using the command above for example, you should receive 1.00 on all metrics. The **-g** argument is for the ground truth file path and **-p** is for the predictions file path.

| EMR (*) | Micro Precision | Micro Recall | Micro F1 | Main Role Accuracy |
|---------|-----------------|--------------|----------|---------------------|
| 1.0000  | 1.0000          | 1.0000       | 1.0000   | 1.0000             |

(*) EMR is the Exact Match Ratio on the fine-grained roles which is the official evaluation metric.

#### Format Checker

The scorer already checks the format and provides logs in case of errors. For those interested in using the format checker only without the scorer, below is an example of usage in python:


```python
from subtask1_scorer import MAIN_ROLES, FINE_GRAINED_ROLES, read_file, check_file_format

gold_file_path = "data/train/EN/subtask-1-annotations.txt"
pred_file_path = "data/train/EN/subtask-1-annotations.txt"

gold_dict = read_file(gold_file_path)
pred_dict = read_file(pred_file_path)

format_errors = check_file_format(gold_dict, pred_dict)
```
