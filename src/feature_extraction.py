import torch
from src.utils import get_device, get_logger
from typing import Dict, List, Tuple, Any
import logging
from src.preprocessing import Preprocessor
import difflib

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    def __init__(self, max_length: int = 512, context_window: int = 128, batch_size: int = 32, preprocessing_config: Dict = None):
        self.max_length = max_length
        self.context_window = context_window
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None
        self.logger = get_logger(__name__)
        self.device = get_device()
        self.preprocessor = Preprocessor()
        self.preprocessing_config = preprocessing_config or {
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
        self.special_tokens = ['[ENT]', '[/ENT]']

    def set_tokenizer_and_model(self, tokenizer, model):
        """Set the tokenizer and model with added special tokens."""
        # Add special tokens for entity markers
        tokenizer.add_special_tokens(
            {'additional_special_tokens': self.special_tokens})
        model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        self.model = model.to(self.device)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess the text using the Preprocessor."""
        return self.preprocessor.preprocess_text(text, **self.preprocessing_config)

    def _adjust_offsets(self, original_text: str, preprocessed_text: str, start_offset: int, end_offset: int) -> Tuple[int, int]:
        """
        Adjust entity offsets after preprocessing with enhanced robustness.
        Uses multiple strategies to find the entity in the preprocessed text.
        """
        original_entity = original_text[start_offset:end_offset]
        preprocessed_entity = self._preprocess_text(original_entity)

        # Strategy 1: Direct match of original entity
        try:
            new_start = preprocessed_text.index(original_entity)
            new_end = new_start + len(original_entity)
            return new_start, new_end
        except ValueError:
            pass

        # Strategy 2: Match preprocessed entity
        try:
            new_start = preprocessed_text.index(preprocessed_entity)
            new_end = new_start + len(preprocessed_entity)
            return new_start, new_end
        except ValueError:
            pass

        # Strategy 3: Fuzzy matching for similar substrings
        # Get the best matching substring
        matcher = difflib.SequenceMatcher(
            None, preprocessed_entity, preprocessed_text)
        match = matcher.find_longest_match(
            0, len(preprocessed_entity), 0, len(preprocessed_text))

        if match.size > len(preprocessed_entity) * 0.8:  # At least 80% match
            new_start = match.b
            new_end = match.b + match.size
            self.logger.debug(
                f"Found fuzzy match for entity: '{original_entity}' -> "
                f"'{preprocessed_text[new_start:new_end]}' (similarity: {match.size/len(preprocessed_entity):.2f})"
            )
            return new_start, new_end

        # Strategy 4: Find closest word boundaries
        words = preprocessed_text.split()
        original_words = original_entity.split()

        for i, word in enumerate(words):
            if any(orig_word in word for orig_word in original_words):
                # Found a partial match, try to expand
                start_idx = sum(len(w) + 1 for w in words[:i])
                end_idx = start_idx + len(word)

                # Check surrounding words for better context
                context_start = max(0, i - 1)
                context_end = min(len(words), i + 2)
                context = " ".join(words[context_start:context_end])

                self.logger.debug(
                    f"Found partial match for entity: '{original_entity}' -> "
                    f"'{context}' at position {start_idx}:{end_idx}"
                )
                return start_idx, end_idx

        # If all strategies fail, log warning and return original offsets
        self.logger.warning(
            f"Could not find entity '{original_entity}' in preprocessed text. "
            f"Using original offsets: {start_offset}:{end_offset}"
        )
        return start_offset, end_offset

    def _find_entity_boundaries(self, text: str, entity_text: str, start_offset: int, end_offset: int) -> Tuple[int, int]:
        """Find the correct entity boundaries in the tokenized text."""
        # Normalize text for comparison
        normalized_entity = entity_text.lower().strip()
        normalized_text = text.lower()

        # First try exact match around the provided offsets
        context_window = self.context_window  # Look within 100 chars before and after
        start_idx = max(0, start_offset - context_window)
        end_idx = min(len(text), end_offset + context_window)
        search_text = normalized_text[start_idx:end_idx]

        # Try to find the entity in the search window
        entity_idx = search_text.find(normalized_entity)
        if entity_idx != -1:
            # Adjust offsets back to original text position
            start_offset = start_idx + entity_idx
            end_offset = start_offset + len(normalized_entity)

        return start_offset, end_offset

    def extract_features(self, text: str, entity_text: str, start_offset: int, end_offset: int) -> Dict[str, Any]:
        """Extract features for a single entity mention."""
        # Find correct entity boundaries
        start_offset, end_offset = self._find_entity_boundaries(
            text, entity_text, start_offset, end_offset)

        # Add entity markers
        marked_text = (
            text[:start_offset] +
            self.special_tokens[0] +
            text[start_offset:end_offset] +
            self.special_tokens[1] +
            text[end_offset:]
        )

        # Tokenize with truncation
        encoding = self.tokenizer(
            marked_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True,
        )

        # Find entity positions in tokenized text
        input_ids = encoding["input_ids"][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        try:
            start_token = tokens.index(self.special_tokens[0])
            end_token = tokens.index(self.special_tokens[1])
        except ValueError:
            # If markers not found, try to find the entity text directly
            entity_tokens = self.tokenizer.tokenize(entity_text)
            for i in range(len(tokens) - len(entity_tokens) + 1):
                if tokens[i:i + len(entity_tokens)] == entity_tokens:
                    start_token = i
                    end_token = i + len(entity_tokens)
                    break
            else:
                # If still not found, use original offsets
                start_token = 0
                # Use first 10 tokens as fallback
                end_token = min(10, len(tokens))

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**encoding)
            embeddings = outputs.last_hidden_state[0]

        # Extract features for the entity span
        entity_embeddings = embeddings[start_token:end_token]
        entity_attention_mask = encoding["attention_mask"][0][start_token:end_token]

        return {
            "input_ids": input_ids.tolist(),
            "attention_mask": encoding["attention_mask"][0].tolist(),
            "entity_position": [start_token, end_token],
            "entity_embeddings": entity_embeddings.tolist(),
            "entity_attention_mask": entity_attention_mask.tolist(),
        }

    def process_batch(self, texts: List[str], entity_spans: List[Tuple[int, int]]) -> List[Dict]:
        """Process a batch with error handling and memory management."""
        features = []

        # Process in smaller batches to manage memory
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_spans = entity_spans[i:i + self.batch_size]

            try:
                batch_features = []
                for text, span in zip(batch_texts, batch_spans):
                    try:
                        feature = self.extract_features(text, *span)
                        batch_features.append(feature)
                    except Exception as e:
                        self.logger.error(
                            f"Error processing example: {str(e)}")
                        # Add None to maintain alignment
                        batch_features.append(None)

                features.extend(batch_features)

                # Clear CUDA cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                self.logger.error(f"Error processing batch: {str(e)}")
                # Add None for each example in the failed batch
                features.extend([None] * len(batch_texts))

        # Remove None values and warn about skipped examples
        valid_features = [f for f in features if f is not None]
        if len(valid_features) < len(texts):
            self.logger.warning(
                f"Skipped {len(texts) - len(valid_features)} examples due to errors")

        return valid_features
