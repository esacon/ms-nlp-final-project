# Entity Framing Classification - Detailed Instructions

## Overview of the Task
Your model needs to:
1. Read news articles
2. Identify provided named entities (e.g., "Martin Luther King Jr.", "United Nations")
3. Classify each entity with:
   - One main role (Protagonist, Antagonist, or Innocent)
   - One or more fine-grained roles (e.g., Martyr, Guardian, Terrorist)

## Project Structure
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

## Model Requirements

### Architecture
1. **Base Model Options**:
   - English: RoBERTa-large
   - Multilingual: XLM-RoBERTa-large
   
2. **Classification Heads**:
   - Main Role Head: 3-way classification (Protagonist, Antagonist, Innocent)
   - Fine-grained Role Head: Multi-label classification (22 possible roles)

3. **Input Processing**:
   - Context Window: 100 tokens on each side of entity mention
   - Special Tokens: [ENT] entity_mention [/ENT] for entity marking
   - Max Sequence Length: 512 tokens

4. **Training Parameters**:
   - Learning Rate: 2e-5
   - Batch Size: 16
   - Epochs: 3
   - Optimizer: AdamW with weight decay
   - Loss Function: 
     * Main Role: CrossEntropyLoss
     * Fine-grained Roles: BCEWithLogitsLoss

## Step-by-Step Process

### 1. Data Preparation

A. **Input Data Structure**
   ```
   data/train/EN/raw-documents/
   ├── article1.txt
   ├── article2.txt
   └── article3.txt
   ```
   Each article looks like:
   ```
   Title: Climate Change Summit Brings Hope
   
   The United Nations hosted a landmark climate summit today. Greta Thunberg criticized world leaders...
   ```

B. **Entity Mentions File**
   ```
   data/train/EN/subtask-1-annotations.txt
   ```
   Contains:
   ```
   article_id    entity_mention    start_offset    end_offset    main_role    fine_grained_roles
   EN_001.txt    United Nations    4              18            Protagonist    Guardian
   EN_001.txt    Greta Thunberg   45             59            Protagonist    Rebel    Martyr
   ```

C. **Data Processing Steps**:
   1. Load and parse article text files (UTF-8 encoding)
   2. Extract entity contexts with 100-token windows
   3. Create input features:
      - Tokenize text with entity markers
      - Convert to model inputs (input_ids, attention_mask)
   4. Create label tensors:
      - Main role: One-hot encoded (3 classes)
      - Fine-grained roles: Multi-hot encoded (22 classes)

### 2. Model Training

A. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

B. **Configure Model**
   Edit `src/configs.yaml`:
   ```yaml
   model:
     name: "xlm-roberta-large"
     learning_rate: 2e-5
     num_epochs: 3
     batch_size: 16
     max_length: 512
     context_window: 100    # text around entity to consider
     gradient_accumulation: 2
     warmup_steps: 500
     weight_decay: 0.01
     label_smoothing: 0.1
   
   training:
     validation_split: 0.1
     early_stopping_patience: 3
     save_best_only: true
     mixed_precision: true
   ```

C. **Train Model**
   ```bash
   python run.py --task train --languages EN --config src/configs.yaml --output_dir models
   ```
   This will:
   1. Load training data
   2. Process articles and entity mentions
   3. Train the model with:
      - Gradient accumulation
      - Mixed precision training
      - Early stopping on validation loss
      - Model checkpointing
   4. Save as `models/model.pt`

### 3. Generate Predictions

A. **Run Inference**
   ```bash
   python run.py --task evaluate -f models/model.pt --languages EN --output_dir predictions
   ```

B. **Expected Output Format**
   Your model should create `predictions/predictions.txt`:
   ```
   article_id    entity_mention    start_offset    end_offset    main_role    fine_grained_roles
   EN_001.txt    United Nations    4              18            Protagonist    Guardian
   EN_001.txt    Greta Thunberg   45             59            Protagonist    Rebel    Martyr
   ```

   Critical requirements:
   - Must be tab-separated
   - File extension must be .txt
   - UTF-8 encoding
   - Exact match of entity mentions from input
   - Valid roles only

### 4. Validate Predictions

A. **Check Format**
   ```python
   from subtask1_scorer import read_file, check_file_format
   
   gold_dict = read_file("data/EN/subtask-1-annotations.txt")
   pred_dict = read_file("predictions/predictions.txt")
   format_errors = check_file_format(gold_dict, pred_dict)
   
   if format_errors:
       print("Fix these errors:", format_errors)
   ```

B. **Valid Values**
   ```python
   # Main roles (exactly one required)
   MAIN_ROLES = ['Protagonist', 'Antagonist', 'Innocent']
   
   # Fine-grained roles (one or more required)
   FINE_GRAINED_ROLES = [
       # Protagonist roles
       'Guardian',    # Protector, defender of values/people
       'Martyr',      # Sacrifices for a cause
       'Peacemaker',  # Mediator, conflict resolver
       'Rebel',       # Challenges authority for good
       'Underdog',    # Faces overwhelming odds
       'Virtuous',    # Morally exemplary
       
       # Antagonist roles
       'Instigator',        # Causes conflicts/problems
       'Conspirator',       # Secret plotter
       'Tyrant',           # Oppressive authority
       'Foreign Adversary', # External threat
       'Traitor',          # Betrays own side
       'Spy',              # Secret agent
       'Saboteur',         # Undermines from within
       'Corrupt',          # Morally compromised
       'Incompetent',      # Harmful through inability
       'Terrorist',        # Uses violence for goals
       'Deceiver',         # Misleads others
       'Bigot',           # Prejudiced/discriminatory
       
       # Innocent roles
       'Forgotten',    # Overlooked/neglected
       'Exploited',    # Used by others
       'Victim',       # Suffers harm
       'Scapegoat'     # Blamed unfairly
   ]
   ```

C. **Validation Steps**:
   1. Check file format (UTF-8, tab-separated)
   2. Verify all required entities are present
   3. Validate main roles (case-sensitive, single value)
   4. Validate fine-grained roles (case-sensitive, one or more)
   5. Check offset consistency with input file

### 5. Run Scorer

A. **Basic Scoring**
   ```bash
   python subtask1_scorer.py -g data/EN/subtask-1-annotations.txt -p predictions/predictions.txt
   ```

B. **Detailed Logging**
   ```bash
   python subtask1_scorer.py -g data/EN/subtask-1-annotations.txt -p predictions/predictions.txt -l
   ```

C. **Understanding Output**
   ```
   EMR    Micro_Precision    Micro_Recall    Micro_F1    Main_Role_Accuracy
   0.7500    0.8000          0.7500         0.7742      0.8000
   ```
   
   Metrics explained:
   - EMR (Exact Match Ratio): % of entities where ALL fine-grained roles match
   - Micro_Precision: % of predicted fine-grained roles that are correct
   - Micro_Recall: % of actual fine-grained roles that were predicted
   - Micro_F1: Harmonic mean of precision and recall
   - Main_Role_Accuracy: % of correct main role predictions

### 6. Common Issues and Solutions

A. **Format Errors**
   - Issue: "Invalid main role 'PROTAGONIST'"
   - Solution: Case sensitive! Use 'Protagonist'

B. **Missing Entities**
   - Issue: "Missing entries in prediction file"
   - Solution: Must predict for ALL entities in input file

C. **Invalid Roles**
   - Issue: "Invalid fine-grained role 'Criminal'"
   - Solution: Use only roles from FINE_GRAINED_ROLES list

D. **Encoding Issues**
   - Issue: "UnicodeDecodeError"
   - Solution: Save files with UTF-8 encoding

### 7. Compare with Baselines

A. **Generate Random Baseline**
   ```bash
   python subtask1_baseline.py \
     --dev_file data/EN/subtask-1-entity-mentions.txt \
     --output_dir subtask1_baselines/EN \
     --baseline_type random
   ```

B. **Generate Majority Baseline**
   ```bash
   python subtask1_baseline.py \
     --train_file data/EN/subtask-1-annotations.txt \
     --dev_file data/EN/subtask-1-entity-mentions.txt \
     --output_dir subtask1_baselines/EN \
     --baseline_type majority
   ```

C. **Compare Scores**
   ```bash
   # Score your model
   python subtask1_scorer.py -g gold.txt -p predictions/predictions.txt
   
   # Score random baseline
   python subtask1_scorer.py -g gold.txt -p subtask1_baselines/EN/baseline_random.txt
   
   # Score majority baseline
   python subtask1_scorer.py -g gold.txt -p subtask1_baselines/EN/baseline_majority.txt
   ```