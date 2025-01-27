Below is a **file-by-file breakdown** of what to consider when implementing a scalable system for **Subtask 1: Entity Framing**. Each file has a **clear purpose** in the pipeline, references **taxonomy** and **config**, and ensures **multilingual** flexibility, **multi-label** classification, and **role** predictions.

---

## 1. `taxonomy.json`
**Purpose**: Defines the **role taxonomy** (both main and fine-grained).  
**Key Points**:  
- Must contain a list/array for `main_roles`: `["Protagonist", "Antagonist", "Innocent"]`.  
- Must contain an object/dict for `fine_grained_roles`, keyed by main role name, each key mapping to a list of fine-grained roles.  
- Makes the system flexible if new roles or languages are added.  
- The model, data loader, and preprocessing scripts parse this file to build label mappings.  

Example snippet:
```json
{
  "main_roles": ["Protagonist", "Antagonist", "Innocent"],
  "fine_grained_roles": {
    "Protagonist": ["Guardian", "Martyr", ...],
    "Antagonist": ["Instigator", "Conspirator", ...],
    "Innocent": ["Forgotten", "Exploited", ...]
  }
}
```

---

## 2. `configs.yaml`
**Purpose**: Centralizes **hyperparameters** and **settings** (model name, learning rates, batch sizes, special tokens, etc.).  
**Key Points**:  
- Contains sections like `model`, `preprocessing`, `training`, `evaluation`.  
- `model`: sets the transformer variant (e.g., `xlm-roberta-large`), epochs, LR, etc.  
- `preprocessing`: context window size, special tokens for wrapping entity mentions, etc.  
- `training`: random seed, validation split, early stopping patience.  
- `evaluation`: defines threshold for multi-label classification and selected metrics (e.g., `exact_match_ratio`).  
- All scripts (training, evaluation, pipeline) read from this file.  

Example snippet:
```yaml
model:
  name: "xlm-roberta-large"
  learning_rate: 2e-5
  ...

preprocessing:
  context_window: 100
  ...

training:
  seed: 42
  ...

evaluation:
  metrics:
    - exact_match_ratio
    - main_role_accuracy
    ...
```

---

## 3. `preprocessing.py`
**Purpose**: Handles **text cleaning** and **context-window extraction** around each entity mention.  
**Key Points**:  
- Takes raw article text plus the offsets for each entity mention.  
- **Wrap** the mention with `[ENT] ... [/ENT]` (from config) to let the model focus on it.  
- Remove undesirable elements (e.g., URLs, emojis) if specified.  
- Return text segments up to a specified max length (`context_window` from config).  
- Must ensure correct **token offsets** for different languages or tokenizers.  
- Usually called inside `data_loader.py` or in a separate pipeline stage.  

---

## 4. `data_loader.py`
**Purpose**: Implements a **PyTorch Dataset** (and DataLoader) for entity-role classification.  
**Key Points**:  
- Reads a tab-separated file with `article_id`, `mention`, `start_offset`, `end_offset`, plus optional `main_role` and `fine_grained_roles` (if training).  
- Uses `preprocessing.py` to retrieve appropriate context text from the article.  
- Uses `taxonomy.json` to map main/fine roles to integer indices for training.  
- For dev/test sets (no roles in file), the dataset will **not** have role labels.  
- Returns dictionary of features suitable for direct input to a transformer (e.g., input_ids, attention_mask, label vectors).  
- Splits data for cross-validation or dev/test sets if needed.  
- Must handle large datasets with efficient reading and minimal memory usage.  

---

## 5. `model.py`
**Purpose**: Defines the **Transformer-based** model with **dual classification heads**.  
**Key Points**:  
1. **Base**: XLM-RoBERTa (or any other model from `configs.yaml`).  
2. **Head A** (Main Role): Outputs exactly one label among `[Protagonist, Antagonist, Innocent]`. Typically a **softmax** over 3 classes.  
3. **Head B** (Fine-Grained Roles): Outputs multi-label predictions. Typically a **sigmoid** with dimension = number of fine-grained roles.  
4. **Loss**: Combines cross-entropy for Head A + BCE (multi-label) for Head B. Weighted or equally combined.  
5. **Forward**: Model returns two sets of logits.  
6. **Forward Pass**:  
   - Input is the tokenized text with special entity tokens.  
   - Output includes hidden states feeding into each classification head.  
- Must handle both **training** (where roles exist) and **inference** (where only text is given).  

---

## 6. `pipeline.py`
**Purpose**: Wraps the **training and validation** loop with model instantiation, data loading, and logging.  
**Key Points**:  
- Instantiates the model using `model.py`.  
- Loads config from `configs.yaml`.  
- Loads data via `data_loader.py`.  
- Implements training steps:  
  - forward pass -> loss -> backward pass -> optimizer update -> checkpointing.  
- Handles **early stopping** (based on dev metric) and logging.  
- Often includes a function `train(...)` and a function `validate(...)`.  
- Optionally supports distributed / multi-GPU if needed.  

---

## 7. `train_model.py`
**Purpose**: **Entry point** for **training** the model on the training dataset.  
**Key Points**:  
- Typically parses command-line args (e.g., `--languages EN,PT`).  
- Calls functions from `pipeline.py` to run the entire training loop.  
- Saves model checkpoints in a dedicated directory (e.g., `models/`).  
- Might log final metrics and store best checkpoint.  
- Ensures seeds are set properly for reproducibility.  

**Basic flow**:
1. Read `configs.yaml`.  
2. Prepare data.  
3. Create model + pipeline.  
4. Train, evaluate, save best model.  

---

## 8. `evaluate_model.py`
**Purpose**: **Entry point** for **evaluating** the model on the dev/test sets.  
**Key Points**:  
- Loads the **trained** model checkpoint.  
- Reads dev/test data from `data_loader.py`.  
- Performs inference to get role predictions.  
- Compares with gold labels (if dev set).  
- Computes metrics: main role accuracy, multi-label F1, exact match ratio, etc.  
- Optionally writes out predictions in required TSV format for submission.  

---

## 9. `run.py`
**Purpose**: **Unified CLI** to handle multiple tasks (train, evaluate, predict, etc.).  
**Key Points**:  
- A single script that can do `python run.py --task train ...` or `python run.py --task evaluate ...`.  
- Internally calls `train_model.py` or `evaluate_model.py` depending on arguments.  
- Makes it easier for end-users to run everything from a single command.  
- Consolidates **common argument parsing** and **reusable logic**.  

---

## Additional Considerations

1. **Scalability**:  
   - Large batch sizes often need gradient accumulation (specified in `configs.yaml`).  
   - Mixed Precision (`fp16`) can speed up training.  
   - Evaluate partial data while training to reduce overhead.  

2. **Multilingual Support**:  
   - Possibly store **multiple** model configs if each language uses a different pretrained checkpoint.  
   - Tokenization might differ (different subword splits). Keep it consistent per language.  

3. **Logging & Checkpointing**:  
   - Decide on how often to log (e.g., `logging_steps`) and how often to checkpoint (`checkpoint_frequency`).  
   - Store minimal checkpoint data (model weights, optimizer state, steps).  

4. **Metrics**:  
   - Fine-grained multi-label accuracy focuses on exact match or micro-F1.  
   - Must always output the single main_role even if official metric emphasizes the fine-grained roles.  

5. **Submission Format**:  
   - For final predictions, create a TSV with: `[article_id, entity_mention, start_offset, end_offset, main_role, fine_grained_roles...]`.  
   - The system must not produce extraneous columns or roles outside the taxonomy.  

With these details per file, you can build a **clean, modular**, and **scalable** system for **Subtask 1: Entity Framing** that meets the competition requirements.