import pytest
import torch

from src.feature_extraction import FeatureExtractor


@pytest.fixture
def feature_extractor():
    """Fixture to initialize the FeatureExtractor instance."""
    return FeatureExtractor()


@pytest.fixture
def sample_texts():
    """Fixture providing sample texts for testing."""
    return ["This is a sample text.", "Another example text for testing.", "Short text"]


def test_extract_features_output_structure(feature_extractor):
    """Test if extract_features returns correct dictionary structure."""
    features = feature_extractor.extract_features("Sample text")

    assert isinstance(features, dict)
    assert all(key in features for key in ["input_ids", "attention_mask", "embeddings"])
    assert isinstance(features["input_ids"], torch.Tensor)
    assert isinstance(features["attention_mask"], torch.Tensor)
    assert isinstance(features["embeddings"], torch.Tensor)


def test_extract_features_dimensions(feature_extractor):
    """Test if output tensors have correct dimensions."""
    text = "Sample text for dimension testing"
    features = feature_extractor.extract_features(text)

    assert features["input_ids"].dim() == 2
    assert features["attention_mask"].dim() == 2
    assert features["embeddings"].dim() == 2
    assert features["embeddings"].shape[1] == 1024  # RoBERTa-large hidden size


def test_max_length_truncation(feature_extractor):
    """Test if long text is properly truncated."""
    long_text = " ".join(["word"] * 1000)
    max_length = 10
    features = feature_extractor.extract_features(long_text, max_length=max_length)

    assert features["input_ids"].shape[1] <= max_length
    assert features["attention_mask"].shape[1] <= max_length


def test_process_batch(feature_extractor, sample_texts):
    """Test batch processing of texts."""
    batch_features = feature_extractor.process_batch(sample_texts)

    assert len(batch_features) == len(sample_texts)
    assert all(isinstance(features, dict) for features in batch_features)
    assert all(
        all(key in features for key in ["input_ids", "attention_mask", "embeddings"])
        for features in batch_features
    )


def test_empty_input(feature_extractor):
    """Test handling of empty input."""
    features = feature_extractor.extract_features("")

    assert features["input_ids"].shape[1] > 0  # Should at least contain special tokens
    assert features["embeddings"].shape[1] == 1024


def test_special_characters(feature_extractor):
    """Test handling of special characters."""
    special_text = "!@#$%^&*()\n\t"
    features = feature_extractor.extract_features(special_text)

    assert features["input_ids"].shape[1] > 0
    assert features["embeddings"].shape[1] == 1024


def test_device_placement(feature_extractor):
    """Test if tensors are on the correct device."""
    features = feature_extractor.extract_features("Test text")
    expected_device = feature_extractor.device

    # Compare only device type, not index
    assert features["input_ids"].device.type == expected_device.type
    assert features["attention_mask"].device.type == expected_device.type
    assert features["embeddings"].device.type == expected_device.type
