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
        labels: Optional[List[Dict]] = None,
        taxonomy: Optional[RoleTaxonomy] = None
    ):
        self.features = features
        self.labels = labels
        self.taxonomy = taxonomy or load_taxonomy()
        self.num_fine_labels = len(get_all_subroles(self.taxonomy))
        self.logger = get_logger(__name__)
        
        # Find max entity length for padding
        self.max_entity_length = max(
            len(feature["entity_embeddings"]) 
            for feature in features 
            if "entity_embeddings" in feature
        )
        
        # Convert labels to tensors if provided
        if self.labels is not None:
            # Create tensors for main and fine-grained labels
            self.main_labels = torch.tensor([label["main_role"] for label in labels], dtype=torch.long)
            
            # Create multi-hot tensor for fine-grained labels
            self.fine_labels = torch.zeros((len(labels), self.num_fine_labels), dtype=torch.float)
            for i, label in enumerate(labels):
                for role_idx in label["fine_roles"]:
                    self.fine_labels[i, role_idx] = 1.0
            
            # Validate label counts
            if len(self.main_labels) != len(self.features):
                self.logger.warning(
                    f"Number of labels ({len(self.main_labels)}) doesn't match number of features ({len(self.features)}). "
                    "Labels will be ignored."
                )
                self.labels = None
                self.main_labels = None
                self.fine_labels = None

    def __len__(self) -> int:
        return len(self.features)

    def pad_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Pad entity embeddings to max_entity_length."""
        current_length = embeddings.size(0)
        if current_length < self.max_entity_length:
            padding = torch.zeros(
                (self.max_entity_length - current_length, embeddings.size(1)),
                dtype=embeddings.dtype,
                device=embeddings.device
            )
            return torch.cat([embeddings, padding], dim=0)
        return embeddings[:self.max_entity_length]

    def __getitem__(self, idx: int) -> Dict:
        """Get a single example from the dataset"""
        if idx >= len(self.features):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.features)} items")
            
        # Convert base features to tensors
        item = {
            "input_ids": torch.tensor(self.features[idx]["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(self.features[idx]["attention_mask"], dtype=torch.long),
            "entity_position": torch.tensor(self.features[idx]["entity_position"], dtype=torch.long)
        }
        
        # Handle entity embeddings with padding
        if "entity_embeddings" in self.features[idx]:
            embeddings = torch.tensor(self.features[idx]["entity_embeddings"], dtype=torch.float)
            item["embeddings"] = self.pad_embeddings(embeddings)
        else:
            item["embeddings"] = None

        # Only add labels if they exist and index is valid
        if self.labels is not None:
            try:
                item["main_labels"] = self.main_labels[idx]
                item["fine_labels"] = self.fine_labels[idx]
            except Exception as e:
                self.logger.warning(f"Error processing labels for index {idx}: {e}")
                pass

        return item


class DataLoader:
    """Class for loading and processing the dataset"""

    def __init__(self, config: Dict, max_articles: int = None):
        self.config = config
        self.logger = get_logger(__name__)
        self.taxonomy = load_taxonomy()
        self._init_role_mappings()
        self.max_articles = max_articles
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

        # Load articles
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
                article_id = filename
                if self.max_articles and len(articles) >= self.max_articles:
                    break
                
                # Read article text
                with open(os.path.join(data_path, filename), "r", encoding="utf-8") as f:
                    text = f.read().strip()
                
                # Preprocess text
                self.logger.debug(f"Preprocessing article {article_id} (length: {len(text)})")
                preprocessed_text = self._preprocess_article(text)
                
                # Get annotations for this article
                annotations = article_annotations.get(article_id, [])
                
                article = Article(
                    id=article_id,
                    text=text,
                    language=language,
                    annotations=annotations,
                    preprocessed_text=preprocessed_text
                )
                articles.append(article)
                
                # Update progress bar description
                pbar.set_postfix({
                    "processed": len(articles),
                    "with_annotations": len([a for a in articles if a.annotations])
                })
                    
        except Exception as e:
            self.logger.error(f"Error loading articles from {data_path}: {str(e)}")
            return []

        self.logger.info(f"\nSummary for {language} ({split}):")
        self.logger.info(f"- Total articles: {len(articles)}")
        self.logger.info(f"- Total annotations: {total_annotations}")
        self.logger.info(f"- Articles with annotations: {len(article_annotations)}")
        self.logger.info("="*50 + "\n")
        return articles

    def prepare_features(
        self,
        articles: List[Article],
        feature_extractor,
    ) -> Tuple[List[Dict], Optional[List[Dict]]]:
        """Prepare features and labels for training/inference"""
        features = []
        labels = []
        has_labels = any(len(article.annotations) > 0 for article in articles)

        # Process each article
        total_entities = sum(len(article.annotations) for article in articles)
        pbar = tqdm(total=total_entities, desc="Extracting features", unit="entities")
        
        for article in articles:
            if self.max_articles and len(features) >= self.max_articles:
                break
            
            for annotation in article.annotations:
                try:
                    # Extract features using preprocessed text
                    feature = feature_extractor.extract_features(
                        text=article.preprocessed_text or article.text,
                        entity_text=annotation.entity_mention,
                        start_offset=annotation.start_offset,
                        end_offset=annotation.end_offset
                    )
                    features.append(feature)

                    # Prepare labels if available
                    if has_labels and annotation.main_role and annotation.fine_grained_roles:
                        label = {
                            "main_role": self.main_role_to_idx[annotation.main_role],
                            "fine_roles": [
                                self.fine_role_to_idx[role]
                                for role in annotation.fine_grained_roles
                                if role in self.fine_role_to_idx
                            ]
                        }
                        labels.append(label)
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        "features": len(features),
                        "labels": len(labels) if has_labels else 0
                    })

                except Exception as e:
                    self.logger.warning(f"Error processing annotation in {article.id}: {e}")
                    continue
        
        pbar.close()
        return features, labels if has_labels else None

    def create_dataloader(
        self,
        features: List[Dict],
        labels: Optional[List[Dict]] = None,
        batch_size: int = 32,
        shuffle: bool = True
    ) -> torch.utils.data.DataLoader:
        """Create a DataLoader for training/inference"""
        dataset = EntityDataset(
            features=features,
            labels=labels,
            taxonomy=self.taxonomy
        )

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.get(
                "num_workers", 0)  # Configurable workers
        )