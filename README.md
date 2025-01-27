# Entity Framing Classification

This project implements a multi-label classification system for identifying entity roles in news articles, supporting both English and Portuguese languages.

## Task Description
The goal of this task is to assign roles to named entities in news articles using a predefined taxonomy of fine-grained roles. The task involves multi-label multi-class text-span classification, where each entity mention can be assigned one or more roles from the taxonomy.

### Taxonomy
The taxonomy consists of three main types of roles: protagonists, antagonists, and innocent. Each of these main roles has several fine-grained roles associated with it.

### Task Requirements
Given a news article and a list of entity mentions with their start and end offsets, participants are required to assign one or more fine-grained roles to each entity mention using the provided taxonomy.

### Data Description
The input data consists of news articles in plain text format (UTF-8) with a title (if available) followed by an empty line, and then the article content. Each article is stored in a separate `.txt` file.

### Gold Labels and Submission Format
The gold labels and submission files should be in a tab-separated format with the following columns:
* `article_id`: the file name of the input article file
* `entity_mention`: the string representing the entity mention
* `start_offset`: the start position of the entity mention
* `end_offset`: the end position of the entity mention
* `main_role`: a string representing the main entity role (one of three values: Protagonist, Antagonist, or Innocent)
* `fine_grained_roles`: a tab-separated list of strings representing the fine-grained role(s) associated with the entity mention

### Example
The following example illustrates the task:
```markdown
Article:
"It is "abundantly clear" that the Met Office cannot scientifically claim to know the current average temperature of the U.K. to a hundredth of a degree centigrade, given that it is using data that has a margin of error of up to 2.5°C, notes the climate journalist Paul Homewood."

Entity Mentions:
* Met Office (start_offset: 10, end_offset: 20)
* Paul Homewood (start_offset: 30, end_offset: 40)

Gold Labels:
* Met Office: Antagonist-[Deceiver]
* Paul Homewood: Protagonist-[Guardian]
```

## Setup

### Requirements
- Python 3.8+
- PyTorch
- Transformers
- CUDA (optional, for GPU support)

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd EntityFraming-Multilingual
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
project/
├── data/                     # Dataset files
│   ├── train/               # Training data
│   └── dev/                 # Development/test data
├── src/                     # Source code
│   ├── model.py            # Model definition
│   ├── pipeline.py         # Training pipeline
│   ├── data_loader.py      # Data loading utilities
│   ├── preprocessing.py    # Text preprocessing
│   ├── taxonomy.json       # Role taxonomy
│   └── configs.yaml        # Configuration
├── models/                  # Saved models
└── predictions/            # Model predictions
```

## Usage

### Training

Train the model using:

```bash
python run.py --task train --languages EN PT --config src/configs.yaml --output_dir models
```

Options:
- `--languages`: Languages to train on (EN, PT, or both)
- `--config`: Path to config file (default: src/configs.yaml)
- `--output_dir`: Where to save the model (default: models)
- `--model_name`: Name of the saved model (default: model.pt)

### Evaluation

Evaluate a trained model using:

```bash
python run.py --task evaluate -f models/model.pt --languages EN --config src/configs.yaml --output_dir predictions
```

The evaluation will:
1. Create a validation split from the training data
2. Generate predictions on the validation set
3. Score the predictions using the official scorer

## Configuration

Key configuration options in `src/configs.yaml`:

```yaml
model:
  name: "roberta-large"
  learning_rate: 2e-5
  num_epochs: 3
  batch_size: 16
  max_length: 512

preprocessing:
  context_window: 100
  min_words_per_line: 4
  remove_urls: true
  remove_emojis: true
```

## Important Notes
* The leaderboard will evaluate predictions for both `main_role` and `fine_grained_roles`, but the official evaluation metric is for `fine_grained_roles`.
* `main_role` should take only one of three values from the 1st level of the taxonomy, while `fine_grained_roles` should take one or more values from the 2nd level of the taxonomy.
* If a participant chooses not to train a model to predict `main_role`, they still need to provide a proper value under `main_role` to pass the format checker code in the scorer.
