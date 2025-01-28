from src.feature_extraction import FeatureExtractor
from transformers import RobertaTokenizerFast, RobertaModel
import unittest
import torch
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)


class TestFeatureExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that will be used for all tests."""
        cls.config = {
            "remove_urls": True,
            "remove_emojis": True,
            "remove_social": True,
            "normalize_unicode": True,
            "min_words_per_line": 4,
            "remove_duplicate_lines": True,
            "remove_short_lines": True,
            "merge_paragraphs": True,
            "clean_text": True
        }

        cls.feature_extractor = FeatureExtractor(
            max_length=512,
            context_window=128,
            batch_size=2,
            preprocessing_config=cls.config
        )

        # Initialize tokenizer and model
        cls.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        cls.model = RobertaModel.from_pretrained("roberta-base")
        cls.feature_extractor.set_tokenizer_and_model(cls.tokenizer, cls.model)

    def test_initialization(self):
        """Test if FeatureExtractor initializes correctly."""
        self.assertEqual(self.feature_extractor.max_length, 512)
        self.assertEqual(self.feature_extractor.context_window, 128)
        self.assertEqual(self.feature_extractor.batch_size, 2)
        self.assertIsNotNone(self.feature_extractor.tokenizer)
        self.assertIsNotNone(self.feature_extractor.model)

    def test_preprocess_text(self):
        """Test text preprocessing."""
        test_text = "Check out https://example.com! 😊 @username"
        processed_text = self.feature_extractor._preprocess_text(test_text)
        self.assertNotIn("https://", processed_text)
        self.assertNotIn("😊", processed_text)
        self.assertNotIn("@username", processed_text)

    def test_extract_features_single(self):
        """Test feature extraction for a single entity."""
        text = "John Smith is the CEO of Tech Corp."
        start_offset = 0
        end_offset = 10  # "John Smith"

        features = self.feature_extractor.extract_features(
            text, start_offset, end_offset)

        # Check if all expected keys are present
        expected_keys = {'input_ids', 'attention_mask',
                         'embeddings', 'entity_position'}
        self.assertEqual(set(features.keys()), expected_keys)

        # Check tensor shapes and basic properties
        # [batch_size=1, seq_length]
        self.assertEqual(len(features['input_ids'].shape), 2)
        self.assertEqual(len(features['attention_mask'].shape), 2)
        # [batch_size=1, seq_length, hidden_size]
        self.assertEqual(len(features['embeddings'].shape), 3)
        # [batch_size=1, 2]
        self.assertEqual(len(features['entity_position'].shape), 2)

        # Check if attention mask matches input_ids length
        self.assertEqual(features['attention_mask'].shape,
                         features['input_ids'].shape)

        # Check if entity position markers are within valid range
        self.assertLess(features['entity_position'][0]
                        [1], features['input_ids'].shape[1])
        # Should be after [CLS] token
        self.assertGreater(features['entity_position'][0][0], 0)

        # Verify the embeddings dimensions
        batch_size, seq_length, hidden_size = features['embeddings'].shape
        self.assertEqual(batch_size, 1)
        self.assertEqual(seq_length, features['input_ids'].shape[1])
        self.assertEqual(hidden_size, 768)  # RoBERTa base hidden size

        # Check if attention mask is binary
        self.assertTrue(torch.all(torch.logical_or(features['attention_mask'] == 0,
                                                   features['attention_mask'] == 1)))

        # Decode the tokenized input to verify entity markers
        tokens = self.tokenizer.convert_ids_to_tokens(features['input_ids'][0])
        entity_start, entity_end = features['entity_position'][0]
        self.assertEqual(tokens[entity_start], '[ENT]')
        self.assertEqual(tokens[entity_end], '[/ENT]')

    def test_process_batch(self):
        """Test batch processing of multiple texts."""
        texts = [
            "John Smith is the CEO.",
            "Mary Johnson leads the team.",
            "The company is headquartered in New York."
        ]
        entity_spans = [
            (0, 10),  # "John Smith"
            (0, 12),  # "Mary Johnson"
            (33, 41)   # "New York"
        ]

        batch_features = self.feature_extractor.process_batch(
            texts, entity_spans)

        # Check if we got features for all examples
        self.assertEqual(len(batch_features), len(texts))

        # Check each feature dict in detail
        for i, features in enumerate(batch_features):
            # Verify structure
            expected_keys = {'input_ids', 'attention_mask',
                             'embeddings', 'entity_position'}
            self.assertEqual(set(features.keys()), expected_keys)

            # Check dimensions
            self.assertEqual(len(features['input_ids'].shape), 2)
            self.assertEqual(len(features['attention_mask'].shape), 2)
            self.assertEqual(len(features['embeddings'].shape), 3)
            self.assertEqual(
                features['embeddings'].shape[-1], 768)  # Hidden size

            # Verify entity markers in tokenized output
            tokens = self.tokenizer.convert_ids_to_tokens(
                features['input_ids'][0])
            entity_start, entity_end = features['entity_position'][0]

            # Check if entity markers are present and in correct position
            self.assertEqual(tokens[entity_start], '[ENT]')
            self.assertEqual(tokens[entity_end], '[/ENT]')

            # Verify the entity text is between the markers
            entity_tokens = tokens[entity_start+1:entity_end]
            entity_text = self.tokenizer.convert_tokens_to_string(
                entity_tokens)

            # Clean up the text for comparison
            entity_text = entity_text.strip()
            original_entity = texts[i][entity_spans[i]
                                       [0]:entity_spans[i][1]].strip()

            # The preprocessed entity text should contain the original entity text
            # (allowing for some preprocessing differences)
            self.assertTrue(
                original_entity.lower() in entity_text.lower() or
                entity_text.lower() in original_entity.lower(),
                f"Entity mismatch: {original_entity} vs {entity_text}"
            )

    def test_adjust_offsets(self):
        """Test offset adjustment after preprocessing."""
        original_text = "John Smith (CEO) leads the company."
        preprocessed_text = self.feature_extractor._preprocess_text(
            original_text)
        start_offset = 0
        end_offset = 10  # "John Smith"

        new_start, new_end = self.feature_extractor._adjust_offsets(
            original_text, preprocessed_text, start_offset, end_offset)

        # Check if new offsets are valid
        self.assertGreaterEqual(new_start, 0)
        self.assertLess(new_end, len(preprocessed_text))
        self.assertLess(new_start, new_end)

    def test_feature_extraction_edge_cases(self):
        """Test feature extraction with edge cases."""
        # Test with very long text
        long_text = "John Smith " * 200  # Create a very long text
        start_offset = 0
        end_offset = 10

        features = self.feature_extractor.extract_features(
            long_text, start_offset, end_offset)
        self.assertLessEqual(
            features['input_ids'].shape[1], self.feature_extractor.max_length)

        # Test with text containing special characters
        special_text = "John Smith (CEO) & VP of R&D @ Tech Corp. [2023]"
        features = self.feature_extractor.extract_features(special_text, 0, 10)
        self.assertIsNotNone(features)

        # Test with entity at the end of text
        end_entity_text = "The CEO is John Smith"
        end_start = len(end_entity_text) - 10
        end_end = len(end_entity_text)
        features = self.feature_extractor.extract_features(
            end_entity_text, end_start, end_end)
        self.assertIsNotNone(features)

        # Test with very short text
        short_text = "John"
        features = self.feature_extractor.extract_features(short_text, 0, 4)
        self.assertIsNotNone(features)


if __name__ == '__main__':
    unittest.main()
