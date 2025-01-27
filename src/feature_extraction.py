import torch
from src.utils import get_device, get_logger
from typing import Dict, List, Tuple
import logging
from src.preprocessing import Preprocessor

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

    def set_tokenizer_and_model(self, tokenizer, model):
        """Set the tokenizer and model with added special tokens."""
        # Add special tokens for entity markers
        tokenizer.add_special_tokens(
            {'additional_special_tokens': ['[ENT]', '[/ENT]']})
        model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        self.model = model.to(self.device)

    def _preprocess_text(self, text: str) -> str:
        """Preprocess the text using the Preprocessor."""
        return self.preprocessor.preprocess_text(text, **self.preprocessing_config)

    def _adjust_offsets(self, original_text: str, preprocessed_text: str, start_offset: int, end_offset: int) -> Tuple[int, int]:
        """Adjust entity offsets after preprocessing."""
        original_entity = original_text[start_offset:end_offset]

        # Find the entity in the preprocessed text
        try:
            new_start = preprocessed_text.index(original_entity)
            new_end = new_start + len(original_entity)
            return new_start, new_end
        except ValueError:
            # If exact match fails, try with preprocessed entity
            preprocessed_entity = self._preprocess_text(original_entity)
            try:
                new_start = preprocessed_text.index(preprocessed_entity)
                new_end = new_start + len(preprocessed_entity)
                return new_start, new_end
            except ValueError:
                # If still not found, return original offsets
                self.logger.warning(
                    f"Could not adjust offsets for entity: {original_entity}")
                return start_offset, end_offset

    def extract_features(self, text: str, start_offset: int, end_offset: int) -> Dict:
        """
        Extract features for a single entity mention, ensuring correct tokenization.
        """
        try:
            if self.tokenizer is None or self.model is None:
                raise ValueError(
                    "Tokenizer and model must be set before feature extraction")

            # Preprocess the text
            preprocessed_text = self._preprocess_text(text)

            # Adjust offsets after preprocessing
            new_start, new_end = self._adjust_offsets(
                text, preprocessed_text, start_offset, end_offset)

            mention = preprocessed_text[new_start:new_end]
            context_start = max(0, new_start - self.context_window)
            context_end = min(len(preprocessed_text),
                              new_end + self.context_window)
            left_context = preprocessed_text[context_start:new_start]
            right_context = preprocessed_text[new_end:context_end]
            marked_text = f"{left_context}[ENT]{mention}[/ENT]{right_context}"

            # Tokenize and find entity markers
            tokens = self.tokenizer.tokenize(marked_text)
            start_marker_idx = end_marker_idx = -1
            for i, token in enumerate(tokens):
                if token == '[ENT]':
                    start_marker_idx = i
                elif token == '[/ENT]':
                    end_marker_idx = i
                    break

            # Fallback: Check with space-aware markers if not found
            if start_marker_idx == -1 or end_marker_idx == -1:
                marked_text = f"{left_context} [ENT] {mention} [/ENT] {right_context}"
                tokens = self.tokenizer.tokenize(marked_text)
                for i, token in enumerate(tokens):
                    if token == '[ENT]':
                        start_marker_idx = i
                    elif token == '[/ENT]':
                        end_marker_idx = i
                        break

            if start_marker_idx == -1 or end_marker_idx == -1:
                self.logger.error("Entity markers not found in tokens")
                raise ValueError("Entity markers not tokenized properly")

            # Calculate available length (accounting for [CLS] and [SEP])
            available_length = self.max_length - 2
            if len(tokens) > available_length:
                # Center truncation around the entity
                center = (start_marker_idx + end_marker_idx) // 2
                half_window = available_length // 2
                start_idx = max(0, center - half_window)
                end_idx = min(len(tokens), center + half_window)
                # Adjust if near boundaries
                if end_idx - start_idx < available_length:
                    if start_idx == 0:
                        end_idx = min(len(tokens), available_length)
                    else:
                        start_idx = max(0, len(tokens) - available_length)
                truncated_tokens = tokens[start_idx:end_idx]
                # Adjust marker positions after truncation
                start_marker_idx = end_marker_idx = -1
                for i, token in enumerate(truncated_tokens):
                    if token == '[ENT]':
                        start_marker_idx = i
                    elif token == '[/ENT]':
                        end_marker_idx = i
                        break
                if start_marker_idx == -1 or end_marker_idx == -1:
                    raise ValueError("Markers lost during truncation")
            else:
                truncated_tokens = tokens

            # Convert tokens to input IDs with special tokens
            input_tokens = [self.tokenizer.cls_token] + \
                truncated_tokens + [self.tokenizer.sep_token]
            input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

            # Pad or truncate
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = [1] * self.max_length
            else:
                pad_len = self.max_length - len(input_ids)
                input_ids += [self.tokenizer.pad_token_id] * pad_len
                attention_mask = [1] * len(input_tokens) + [0] * pad_len

            # Convert to tensors and move to device
            input_ids = torch.tensor(
                [input_ids], dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(
                [attention_mask], dtype=torch.long, device=self.device)
            entity_position = torch.tensor(
                [[start_marker_idx + 1, end_marker_idx + 1]], dtype=torch.long, device=self.device)

            # Get embeddings
            with torch.no_grad():
                try:
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask)
                    embeddings = outputs.last_hidden_state
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        self.logger.warning("GPU OOM, attempting CPU fallback")
                        # Move to CPU as fallback
                        input_ids = input_ids.cpu()
                        attention_mask = attention_mask.cpu()
                        self.model = self.model.cpu()
                        outputs = self.model(
                            input_ids=input_ids, attention_mask=attention_mask)
                        embeddings = outputs.last_hidden_state
                        self.model = self.model.to(self.device)

            return {
                'input_ids': input_ids.cpu(),
                'attention_mask': attention_mask.cpu(),
                'embeddings': embeddings.cpu(),
                'entity_position': entity_position.cpu()
            }

        except Exception as e:
            self.logger.error(f"Error in feature extraction: {str(e)}")
            raise

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
