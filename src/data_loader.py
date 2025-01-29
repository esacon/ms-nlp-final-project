import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from src.utils import get_logger
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from src.taxonomy import (
    MainRole, FineGrainedRole, RoleTaxonomy,
    load_taxonomy, validate_role, TaxonomyError,
    get_all_subroles, MAIN_ROLES
)
from src.preprocessing import Preprocessor
import numpy as np

logger = get_logger(__name__)


@dataclass
class EntityAnnotation:
    article_id: str
    entity_mention: str
    start_offset: int
    end_offset: int
    main_role: Optional[MainRole] = None
    fine_grained_roles: Optional[List[FineGrainedRole]] = None


@dataclass
class Article:
    """Class for storing article data"""
    id: str
    text: str
    language: str
    annotations: List[EntityAnnotation]
    preprocessed_text: Optional[str] = None


class EntityDataset(Dataset):
    """Dataset class for entity role classification"""

    def __init__(
        self,
        features: List[Dict],
        labels: Optional[Dict] = None,
        taxonomy: Optional[RoleTaxonomy] = None
    ):
        self.features = features
        self.labels = labels
        self.taxonomy = taxonomy or load_taxonomy()
        self.num_fine_labels = len(get_all_subroles(self.taxonomy))
        self.logger = get_logger(__name__)
        
        # Validate that if we have labels, we have them for all features
        if self.labels is not None:
            if len(self.labels["main_labels"]) != len(self.features):
                self.logger.warning(
                    f"Number of labels ({len(self.labels['main_labels'])}) doesn't match number of features ({len(self.features)}). "
                    "Labels will be ignored."
                )
                self.labels = None

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single example from the dataset"""
        if idx >= len(self.features):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.features)} items")
            
        # Get features for this item
        feature = self.features[idx]
        
        # Convert features to numpy arrays for consistent handling
        item = {
            "input_ids": np.array(feature["input_ids"]),
            "attention_mask": np.array(feature["attention_mask"]),
            "entity_position": np.array(feature["entity_position"]),
        }
        
        # Add embeddings if they exist
        if "entity_embeddings" in feature:
            item["embeddings"] = np.array(feature["entity_embeddings"])

        # Only add labels if they exist and index is valid
        if self.labels is not None:
            try:
                item["main_labels"] = np.array(self.labels["main_labels"][idx])
                item["fine_labels"] = np.array(self.labels["fine_labels"][idx])
            except Exception as e:
                self.logger.warning(f"Error processing labels for index {idx}: {e}")
                pass
        
        return item

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function to handle variable length sequences.
        
        Args:
            batch: List of dictionaries containing features and labels
            
        Returns:
            Dictionary with batched tensors
        """
        # Check if we have labels in the batch
        has_labels = "main_labels" in batch[0] and "fine_labels" in batch[0]
        
        # Get max sequence length in this batch
        max_len = max(len(item["input_ids"]) for item in batch)
        
        # Initialize tensors
        batch_size = len(batch)
        input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        entity_positions = torch.zeros((batch_size, 2), dtype=torch.long)
        
        # Fill tensors
        for i, item in enumerate(batch):
            # Get length of this sequence
            seq_len = len(item["input_ids"])
            
            # Convert to tensors and pad
            input_ids[i, :seq_len] = torch.tensor(item["input_ids"], dtype=torch.long)
            attention_mask[i, :seq_len] = torch.tensor(item["attention_mask"], dtype=torch.long)
            entity_positions[i] = torch.tensor(item["entity_position"], dtype=torch.long)
        
        # Create output dictionary
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "entity_position": entity_positions
        }
        
        # Add labels if they exist
        if has_labels:
            main_labels = torch.tensor([item["main_labels"] for item in batch], dtype=torch.long)
            fine_labels = torch.tensor([item["fine_labels"] for item in batch], dtype=torch.float)
            output["main_labels"] = main_labels
            output["fine_labels"] = fine_labels
        
        return output


class DataLoader:
    """Class for loading and processing the dataset"""

    def __init__(self, config: Dict, max_articles: int = None):
        """Initialize the DataLoader with configuration.
        
        Args:
            config: Configuration dictionary containing paths and settings
            max_articles: Maximum number of articles to process (for debugging)
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.taxonomy = load_taxonomy()
        self.max_articles = max_articles
        self._init_role_mappings()

        # Initialize preprocessor with config
        self.preprocessor = Preprocessor()
        self.preprocessing_config = config.get("preprocessing", {
            "remove_urls": True,
            "remove_emojis": True,
            "remove_social": True,
            "normalize_unicode": True,
            "min_words_per_line": 4,
            "remove_duplicate_lines": True,
            "remove_short_lines": True,
            "merge_paragraphs": True,
            "clean_text": True
        })

    def _init_role_mappings(self) -> None:
        """Initialize role to index mappings from taxonomy"""
        # Create main role mapping
        self.main_role_to_idx = {
            role: idx for idx, role in enumerate(MAIN_ROLES)}
        self.idx_to_main_role = {idx: role for role,
                                 idx in self.main_role_to_idx.items()}

        # Create fine-grained role mapping
        self.fine_role_to_idx = {}
        self.idx_to_fine_role = {}
        idx = 0
        for main_role in MAIN_ROLES:
            for fine_role in self.taxonomy[main_role]:
                self.fine_role_to_idx[fine_role] = idx
                self.idx_to_fine_role[idx] = fine_role
                idx += 1

    def _preprocess_article(self, text: str) -> str:
        """Preprocess article text"""
        return self.preprocessor.preprocess_text(text, **self.preprocessing_config)

    def _parse_annotation(self, parts: List[str], filename: str) -> EntityAnnotation:
        """Parse a single annotation line into an EntityAnnotation object"""
        if len(parts) < 5:
            raise ValueError(f"Insufficient fields in annotation: {parts}")

        try:
            main_role = parts[4]
            if main_role not in MAIN_ROLES:
                raise ValueError(f"Invalid main role: {main_role}")

            fine_roles = parts[5:] if len(parts) > 5 else []
            # Validate roles against taxonomy
            validate_role(self.taxonomy, main_role, fine_roles)

            return EntityAnnotation(
                article_id=parts[0],
                entity_mention=parts[1],
                start_offset=int(parts[2]),
                end_offset=int(parts[3]),
                main_role=main_role,
                fine_grained_roles=fine_roles
            )
        except (ValueError, TaxonomyError) as e:
            raise ValueError(f"Error in {filename}: {str(e)}")

    def load_articles(self, data_dir: str, language: str, split: str = "train") -> List[Article]:
        """Load articles from the data directory
        
        Args:
            data_dir: Root data directory
            language: Language code (EN or PT)
            split: Data split (train, dev, or test)
            
        Returns:
            List of Article objects
        """
        articles = []
        
        self.logger.info(f"\n{'='*50}\nLoading {language} articles from {split} split\n{'='*50}")
        
        # Handle different directory structures based on split
        if split == "train":
            documents_dir = "raw-documents"
            annotations_file = "subtask-1-annotations.txt"
        elif split == "dev":
            documents_dir = "subtask-1-documents"
            annotations_file = "subtask-1-annotations.txt"
        else:  # test
            documents_dir = "subtask-1-documents"
            annotations_file = "subtask-1-entity-mentions.txt"
            
        # Construct paths
        data_path = os.path.join(data_dir, split, language, documents_dir)
        ann_path = os.path.join(data_dir, split, language, annotations_file)
        
        self.logger.info(f"Looking for documents in: {data_path}")
        self.logger.info(f"Looking for annotations in: {ann_path}")
        
        if not os.path.exists(data_path):
            self.logger.error(f"Directory not found: {data_path}")
            return []

        try:
            # First, load all annotations into a dictionary for faster lookup
            article_annotations: Dict[str, List[EntityAnnotation]] = {}
            total_annotations = 0
            
            if os.path.exists(ann_path):
                self.logger.info("Loading annotations...")
                with open(ann_path, "r", encoding="utf-8") as f:
                    # Count total lines for progress bar
                    total_lines = sum(1 for _ in f)
                    f.seek(0)  # Reset file pointer
                    
                    # Create progress bar for annotations
                    pbar = tqdm(enumerate(f, 1), total=total_lines, 
                              desc=f"Loading {split} annotations", 
                              unit="annotations")
                    
                    for line_num, line in pbar:
                        try:
                            parts = line.strip().split("\t")
                            article_id = parts[0]
                            
                            # Handle different annotation formats for train/dev vs test
                            if split in ["train", "dev"]:
                                annotation = self._parse_annotation(parts, ann_path)
                                self.logger.debug(f"Parsed annotation for {article_id}: {annotation.entity_mention} "
                                               f"[{annotation.main_role}: {annotation.fine_grained_roles}]")
                            else:
                                # Test data only has entity mentions
                                annotation = EntityAnnotation(
                                    article_id=article_id,
                                    entity_mention=parts[1],
                                    start_offset=int(parts[2]),
                                    end_offset=int(parts[3])
                                )
                                self.logger.debug(f"Parsed mention for {article_id}: {annotation.entity_mention}")
                            
                            # Clean up article ID by removing file extension if present
                            article_id = os.path.splitext(article_id)[0]
                            
                            if article_id not in article_annotations:
                                article_annotations[article_id] = []
                            article_annotations[article_id].append(annotation)
                            total_annotations += 1
                            
                            # Update progress bar description
                            if line_num % 1000 == 0:
                                pbar.set_postfix({"articles": len(article_annotations)})
                                
                        except Exception as e:
                            self.logger.warning(f"Error parsing annotation at line {line_num} in {ann_path}: {e}")
                            continue
                
                self.logger.info(f"Loaded {total_annotations} annotations for {len(article_annotations)} articles")

            # Then load and process articles
            self.logger.info("\nLoading and preprocessing articles...")
            
            # Get list of files and create progress bar
            article_files = [f for f in os.listdir(data_path) if f.endswith(".txt")]
            pbar = tqdm(article_files, desc=f"Processing {split} articles", unit="articles")
            
            for filename in pbar:
                if self.max_articles and len(articles) >= self.max_articles:
                    break
                
                article_id = os.path.splitext(filename)[0]  # Remove .txt extension
                
                try:
                    # Read article text
                    with open(os.path.join(data_path, filename), "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    
                    # Preprocess text
                    self.logger.debug(f"Preprocessing article {article_id} (length: {len(text)})")
                    preprocessed_text = self._preprocess_article(text)
                    
                    # Get annotations for this article
                    annotations = article_annotations.get(article_id, [])
                    
                    # Only add articles that have annotations
                    if annotations:
                        article = Article(
                            id=article_id,
                            text=text,
                            language=language,
                            annotations=annotations,
                            preprocessed_text=preprocessed_text
                        )
                        articles.append(article)
                        self.logger.debug(f"Added article {article_id} with {len(annotations)} annotations")
                    else:
                        self.logger.debug(f"Skipping article {article_id} - no annotations found")
                    
                    # Update progress bar description
                    pbar.set_postfix({
                        "processed": len(articles),
                        "with_annotations": len([a for a in articles if a.annotations])
                    })
                except Exception as e:
                    self.logger.error(f"Error processing article {filename}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error loading articles from {data_path}: {str(e)}")
            return []

        self.logger.info(f"\nSummary for {language} ({split}):")
        self.logger.info(f"- Total articles processed: {len(article_files)}")
        self.logger.info(f"- Articles with annotations: {len(articles)}")
        self.logger.info(f"- Total annotations: {total_annotations}")
        self.logger.info("="*50 + "\n")
        
        return articles

    def prepare_features(
        self,
        articles: List[Article],
        feature_extractor,
    ) -> Tuple[List[Dict], Optional[Dict]]:
        """Prepare features and labels for training/inference"""
        features = []
        main_labels = []
        fine_labels = []
        has_labels = any(len(article.annotations) > 0 for article in articles)

        # Process each article
        total_entities = sum(len(article.annotations) for article in articles)
        pbar = tqdm(total=total_entities, desc="Extracting features", unit="entities")
        
        for article in articles:
            for annotation in article.annotations:
                try:
                    # Get the entity text from the article using the annotation offsets
                    entity_text = article.text[annotation.start_offset:annotation.end_offset]
                    
                    # Extract features using preprocessed text
                    feature = feature_extractor.extract_features(
                        text=article.preprocessed_text or article.text,  # Fallback to original if needed
                        entity_text=entity_text,
                        start_offset=annotation.start_offset,
                        end_offset=annotation.end_offset
                    )
                    features.append(feature)

                    # Prepare labels if available
                    if has_labels and annotation.main_role and annotation.fine_grained_roles:
                        main_labels.append(self.main_role_to_idx[annotation.main_role])
                        fine_label = [
                            self.fine_role_to_idx[role]
                            for role in annotation.fine_grained_roles
                            if role in self.fine_role_to_idx
                        ]
                        fine_labels.append(fine_label)
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        "features": len(features),
                        "labels": len(main_labels) if has_labels else 0
                    })
                except Exception as e:
                    self.logger.warning(f"Error processing annotation in {article.id}: {e}")
                    continue
        
        pbar.close()
        
        # Return features and labels in the correct format
        if has_labels:
            labels = {
                "main_labels": main_labels,
                "fine_labels": fine_labels
            }
            return features, labels
        return features, None

    def create_dataset(
        self,
        features: List[Dict],
        labels: Optional[Dict] = None,
        taxonomy: Optional[Dict] = None
    ) -> EntityDataset:
        """Create a dataset from features and labels.
        
        Args:
            features: List of extracted features
            labels: Optional dictionary containing main_labels and fine_labels
            taxonomy: Optional taxonomy dictionary
            
        Returns:
            EntityDataset object
        """
        return EntityDataset(
            features=features,
            labels=labels,
            taxonomy=taxonomy or self.taxonomy
        )

    def create_dataloader(
        self,
        features: List[Dict],
        labels: Optional[Dict] = None,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        """Create a PyTorch DataLoader from features and labels.
        
        Args:
            features: List of extracted features
            labels: Optional dictionary containing main_labels and fine_labels
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            
        Returns:
            PyTorch DataLoader
        """
        # Create dataset
        dataset = self.create_dataset(features, labels)
        
        # Create dataloader with custom collate function
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda x: dataset.collate_fn(x)
        )

    def load_data(self, split: str = "train") -> List[Article]:
        """Load data for the specified split.
        
        Args:
            split: Data split to load ('train', 'dev', or 'test')
            
        Returns:
            List of Article objects
        """
        languages = self.config["languages"]["default"].split()
        all_articles = []
        
        # Get the correct data directory based on split
        data_dir_key = f"{split}_data_dir"
        data_dir = self.config["paths"].get(data_dir_key)
        
        if not data_dir:
            self.logger.error(f"No data directory configured for split '{split}'")
            return []
        
        for language in languages:
            articles = self.load_articles(
                data_dir=data_dir,
                language=language,
                split=split
            )
            if self.max_articles:
                articles = articles[:self.max_articles]
            all_articles.extend(articles)
            
        return all_articles