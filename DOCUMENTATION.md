# SemEval 2025 Task 10 - Subtask 1: Entity Framing
## Project Documentation

### Task Overview
This project addresses Subtask 1 of SemEval 2025 Task 10, focusing on Entity Framing in multilingual news articles. The challenge is part of a broader task on "Multilingual Characterization and Extraction of Narratives from Online News."

#### Core Task Description
- **Objective**: Automatically identify and classify the roles of named entities in news articles
- **Task Type**: Multi-label, multi-class text-span classification
- **Domains**: 
  - Ukraine-Russia War
  - Climate Change
- **Languages Supported**:
  - English (EN)
  - Portuguese (PT)
  - (Task also includes Bulgarian, Hindi, and Russian - not implemented yet)

### Classification Framework

#### Role Taxonomy
1. **Main Roles** (First Level)
   - Protagonist
   - Antagonist
   - Innocent

2. **Fine-grained Roles** (Second Level)
   - Multiple sub-roles under each main category
   - Example:
     - Protagonist: [Guardian]
     - Antagonist: [Deceiver]

### Data Format

#### Input Format
1. **Article Files**
   - Format: Plain text (UTF-8)
   - Structure:
     - Title (if available)
     - Empty line
     - Article content
   - Location: Separate .txt files

2. **Entity Information**
   - Named entity mentions
   - Start and end offsets in text
   - Original text span

#### Output Format (Predictions)
Tab-separated format containing:
1. article_id (filename)
2. entity_mention (text)
3. start_offset (integer)
4. end_offset (integer)
5. main_role (single label)
6. fine_grained_roles (multiple labels possible)

### Data Organization

#### Directory Structure
```
data/
├── train/                    # Training data
│   ├── EN/                  # English data
│   │   ├── raw-documents/   # Text files of articles
│   │   └── subtask-1-annotations.txt  # Entity annotations with roles
│   └── PT/                  # Portuguese data
│       ├── raw-documents/   # Text files of articles
│       └── subtask-1-annotations.txt  # Entity annotations with roles
├── dev/                     # Development data
│   ├── EN/                  # English evaluation data
│   │   ├── subtask-1-documents/     # Text files of articles
│   │   └── subtask-1-entity-mentions.txt  # Entity mentions without roles
│   └── PT/                  # Portuguese evaluation data
│       ├── subtask-1-documents/     # Text files of articles
│       └── subtask-1-entity-mentions.txt  # Entity mentions without roles
└── test/                    # Test data
    ├── EN/                  # English test data
    │   ├── subtask-1-documents/     # Text files of articles
    │   └── subtask-1-entity-mentions.txt  # Entity mentions without roles
    └── PT/                  # Portuguese test data
        ├── subtask-1-documents/     # Text files of articles
        └── subtask-1-entity-mentions.txt  # Entity mentions without roles
```

#### Task Format and Requirements

##### Annotation Format
The task uses tab-separated files with the following format:
```
article_id    entity_mention    start_offset    end_offset    main_role    fine_grained_roles
```

Where:
- `article_id`: File name of the input article (e.g., "EN_10001.txt")
- `entity_mention`: String representing the entity (e.g., "Martin Luther King Jr.")
- `start_offset`: Start position of mention (e.g., "10")
- `end_offset`: End position of mention (e.g., "32")
- `main_role`: One of three first-level taxonomy values (Protagonist/Antagonist/Innocent)
- `fine_grained_roles`: Tab-separated list of second-level taxonomy roles

Example annotations:
```
EN_10001.txt    Martin Luther King Jr.    10    32    Protagonist    Martyr
EN_10002.txt    Mahatma Gandhi           12    27    Protagonist    Martyr    Rebel
EN_10003.txt    ISIS                     4     8     Antagonist     Terrorist    Deceiver
```

##### Important Notes
1. **Entity Recognition**: Named-Entity recognition is not required. The dev set provides `subtask-1-entity-mentions.txt` with all entity mentions and their offsets.

2. **Role Prediction Requirements**:
   - `main_role`: Must be exactly one of the first-level taxonomy values
   - `fine_grained_roles`: Can be one or more second-level taxonomy values
   - Both fields must be provided in submissions, even if only training for fine-grained roles

3. **Evaluation**:
   - Official metric focuses on fine_grained_roles accuracy
   - Leaderboard shows both main_role and fine_grained_roles performance
   - Format checker requires valid main_role values

### Technical Implementation

#### Model Architecture
- Base Model: RoBERTa-large
- Modifications for multi-label classification:
  - Custom classification head with sigmoid activation
  - Separate heads for main_role (3 classes) and fine_grained_roles (multi-label)
- Multilingual support through XLM-RoBERTa for PT language
- Context window approach for entity-centric classification

#### Project Structure
```
project/
├── data/
│   ├── scorers/
│   │   ├── README.md
│   │   ├── subtask1_scorer.py
│   │   └── subtask1_baseline.py
│   ├── train/
│   │   ├── EN/
│   │   │   ├── raw-documents/
│   │   │   └── subtask-1-annotations.txt
│   │   └── PT/
│   │       ├── raw-documents/
│   │       └── subtask-1-annotations.txt
│   ├── dev/
│   │   ├── EN/
│   │   │   ├── subtask-1-documents/
│   │   │   ├── subtask-1-annotations.txt
│   │   │   └── subtask-1-entity-mentions.txt
│   │   └── PT/
│   │       ├── subtask-1-documents/
│   │       ├── subtask-1-annotations.txt
│   │       └── subtask-1-entity-mentions.txt
│   └── test/
│       ├── EN/
│       │   ├── subtask-1-documents/
│       │   └── subtask-1-entity-mentions.txt
│       └── PT/
│           ├── subtask-1-documents/
│           └── subtask-1-entity-mentions.txt
├── src/
│   ├── model.py
│   ├── pipeline.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── taxonomy.json
│   └── configs.yaml
├── models/
├── predictions/
├── requirements.txt
├── README.md
└── DOCUMENTATION.md
```

#### Key Components

1. **Preprocessing (preprocessing.py)**
   - Context window handling with entity-centered text spans
   - Text cleaning and normalization
   - Entity span processing with special tokens ([ENT] entity [/ENT])
   - Multilingual support with language-specific tokenization
   - Handling of document structure (title, content separation)

2. **Data Loading (data_loader.py)**
   - Custom PyTorch Dataset implementation
   - Dynamic batch creation with padding
   - Entity mention handling with offset mapping
   - Efficient memory management for large datasets
   - Cross-validation split functionality

3. **Model (model.py)**
   - Transformer-based architecture with dual classification heads
   - Custom loss function combining BCE and CrossEntropy
   - Gradient accumulation for large batch simulation
   - Language-specific model selection and handling
   - Attention masking for entity-focused learning

4. **Training Pipeline (pipeline.py)**
   - Training loop with early stopping
   - Learning rate scheduling
   - Mixed precision training
   - Model checkpointing and versioning
   - Validation metrics tracking
   - Distributed training support

### Configuration

#### Model Configuration (configs.yaml)
```yaml
model:
  name: "roberta-large"
  learning_rate: 2e-5
  num_epochs: 3
  batch_size: 16
  max_length: 512
  gradient_accumulation_steps: 2
  warmup_steps: 500
  weight_decay: 0.01
  fp16: true
  
preprocessing:
  context_window: 100
  min_words_per_line: 4
  remove_urls: true
  remove_emojis: true
  special_tokens:
    entity_start: "[ENT]"
    entity_end: "[/ENT]"
  
training:
  seed: 42
  validation_split: 0.1
  early_stopping_patience: 3
  checkpoint_frequency: 1000
  logging_steps: 100
  
evaluation:
  metrics:
    - exact_match_ratio
    - main_role_accuracy
    - fine_grained_f1
  threshold: 0.5
```

### Usage Instructions

#### Training
```bash
# Option 1: Direct training script
python train_model.py --languages EN --output_dir models

# Option 2: Main runner
python run.py --task train --languages EN
```

#### Evaluation
```bash
# Option 1: Evaluation script
python evaluate_model.py \
  --model_path models/model.pt \
  --train_dir data/train \
  --languages EN

# Option 2: Main runner
python run.py --task evaluate -f models/model.pt --languages EN
```

### Evaluation Metrics

The official evaluation measure is Exact Match Ratio:
- Measures proportion of samples where all labels are correctly predicted
- Considers both main_role and fine_grained_roles
- Primary focus on fine_grained_roles accuracy

### Important Notes

1. **Submission Requirements**
   - Must provide both main_role and fine_grained_roles
   - main_role must be one of three predefined values
   - fine_grained_roles can be multiple values

2. **Evaluation Focus**
   - Official metric focuses on fine_grained_roles
   - main_role must still be provided for format validation

3. **Language Support**
   - Current implementation supports EN and PT
   - Framework ready for extension to other languages

### Timeline and Deadlines

Key dates for SemEval 2025 Task 10:
- Training data release: September-December 2024
- Development dataset: November 2024
- Test phase: January 2025
- System paper submission: February 2025

### References

Official task website: https://propaganda.math.unipd.it/semeval2025task10/ 

### Performance Optimization

#### Training Optimizations
- Mixed precision training (FP16)
- Gradient accumulation for effective batch size increase
- Learning rate scheduling with warmup
- Early stopping based on validation metrics

#### Inference Optimizations
- Batch inference with dynamic batching
- Model quantization for production
- Caching of preprocessed contexts
- Parallel processing for multiple languages