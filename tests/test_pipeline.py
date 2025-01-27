import unittest
import os
import torch
from transformers import RobertaTokenizerFast, RobertaModel
from src.data_loader import DataLoader, EntityDataset
from src.feature_extraction import FeatureExtractor
from src.preprocessing import Preprocessor

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment with sample config and models"""
        cls.config = {
            "preprocessing": {
                "remove_urls": True,
                "remove_emojis": True,
                "remove_social": True,
                "normalize_unicode": True,
                "min_words_per_line": 4,
                "remove_duplicate_lines": True,
                "remove_short_lines": True,
                "merge_paragraphs": True,
                "clean_text": True
            },
            "model": {
                "name": "roberta-base",
                "max_length": 512,
                "context_window": 128,
                "batch_size": 32
            }
        }
        
        # Initialize tokenizer and model
        cls.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        cls.model = RobertaModel.from_pretrained("roberta-base")
        
        # Initialize components
        cls.data_loader = DataLoader(cls.config)
        cls.feature_extractor = FeatureExtractor(
            max_length=cls.config["model"]["max_length"],
            context_window=cls.config["model"]["context_window"],
            batch_size=cls.config["model"]["batch_size"],
            preprocessing_config=cls.config["preprocessing"]
        )
        cls.feature_extractor.set_tokenizer_and_model(cls.tokenizer, cls.model)

    def test_load_and_preprocess_articles(self):
        """Test loading and preprocessing articles"""
        # Test for both languages and splits
        for split in ["train", "dev", "test"]:
            for language in ["EN", "PT"]:
                articles = self.data_loader.load_articles("data", language, split)
                
                self.assertTrue(len(articles) > 0, f"No articles loaded for {language} ({split})")
                
                for article in articles:
                    # Check article structure
                    self.assertIsNotNone(article.text)
                    self.assertIsNotNone(article.preprocessed_text)
                    self.assertEqual(article.language, language)
                    
                    # Verify preprocessing effects
                    self.assertNotEqual(article.text, article.preprocessed_text)
                    self.assertFalse(any(url in article.preprocessed_text 
                                       for url in ["http://", "https://", "www."]))
                    
                    # Check annotations based on split
                    if split == "train":
                        # Training data should have role annotations
                        for ann in article.annotations:
                            self.assertIsNotNone(ann.main_role)
                            self.assertIsNotNone(ann.fine_grained_roles)
                    else:
                        # Dev/test data should only have entity mentions
                        for ann in article.annotations:
                            self.assertIsNone(ann.main_role)
                            self.assertIsNone(ann.fine_grained_roles)

    def test_feature_extraction(self):
        """Test feature extraction from articles"""
        # Test with training data
        articles = self.data_loader.load_articles("data", "EN", "train")
        self.assertTrue(len(articles) > 0, "No articles loaded for testing")
        
        # Get features
        features, labels = self.data_loader.prepare_features(
            articles[:2],  # Test with first 2 articles
            self.feature_extractor
        )
        
        # Check features structure
        self.assertTrue(len(features) > 0, "No features extracted")
        for feature in features:
            # Check required fields
            self.assertIn("input_ids", feature)
            self.assertIn("attention_mask", feature)
            self.assertIn("entity_position", feature)
            self.assertIn("embeddings", feature)
            
            # Check tensor shapes
            self.assertEqual(feature["input_ids"].size(1), self.config["model"]["max_length"])
            self.assertEqual(feature["attention_mask"].size(1), self.config["model"]["max_length"])
            self.assertEqual(feature["entity_position"].size(1), 2)  # Start and end positions

    def test_entity_handling(self):
        """Test entity mention handling and offset adjustment"""
        # Create a sample article with known entities
        sample_text = """
        The United Nations hosted a climate summit today.
        Greta Thunberg criticized world leaders for their inaction.
        """
        
        preprocessor = Preprocessor()
        preprocessed_text = preprocessor.preprocess_text(sample_text)
        
        # Test entity extraction with known offsets
        entities = [
            (sample_text.find("United Nations"), sample_text.find("United Nations") + len("United Nations")),
            (sample_text.find("Greta Thunberg"), sample_text.find("Greta Thunberg") + len("Greta Thunberg"))
        ]
        
        for start, end in entities:
            feature = self.feature_extractor.extract_features(
                text=preprocessed_text,
                start_offset=start,
                end_offset=end
            )
            
            # Verify entity markers in tokenization
            tokens = self.tokenizer.convert_ids_to_tokens(feature["input_ids"][0])
            entity_start = feature["entity_position"][0][0]
            entity_end = feature["entity_position"][0][1]
            
            # Check entity markers are present
            self.assertEqual(tokens[entity_start], "<e>")
            self.assertEqual(tokens[entity_end], "</e>")
            
            # Check entity is between markers
            entity_tokens = tokens[entity_start+1:entity_end]
            self.assertTrue(len(entity_tokens) > 0)

    def test_dataloader_creation(self):
        """Test creation of PyTorch DataLoader"""
        # Test with both training and dev data
        for split in ["train", "dev"]:

            articles = self.data_loader.load_articles("data", "EN", split)
            features, labels = self.data_loader.prepare_features(articles, self.feature_extractor)
            
            # Create dataloader
            batch_size = 2
            dataloader = self.data_loader.create_dataloader(
                features=features,
                labels=labels,
                batch_size=batch_size,
                shuffle=True
            )
            
            # Check dataloader properties
            self.assertEqual(type(dataloader).__name__, "DataLoader")
            
            # Verify dataset properties
            dataset = dataloader.dataset
            self.assertIsInstance(dataset, EntityDataset)
            self.assertTrue(len(dataset) > 0, f"Dataset is empty for {split} split")
            
            # Try getting a batch
            try:
                batch = next(iter(dataloader))
                self.assertIsInstance(batch, dict)
                
                # Check common fields that should always be present
                self.assertIn("input_ids", batch)
                self.assertIn("attention_mask", batch)
                self.assertIn("entity_position", batch)
                
                # Check labels based on split
                if split == "train":
                    self.assertIn("main_labels", batch, "Training batch should have main_labels")
                    self.assertIn("fine_labels", batch, "Training batch should have fine_labels")
                else:
                    self.assertNotIn("main_labels", batch, f"{split} batch should not have main_labels")
                    self.assertNotIn("fine_labels", batch, f"{split} batch should not have fine_labels")
                    
            except Exception as e:
                self.fail(f"Failed to get batch from {split} dataloader: {str(e)}")

if __name__ == "__main__":
    unittest.main() 