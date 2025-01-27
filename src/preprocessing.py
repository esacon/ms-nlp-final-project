import re
import emoji
from nltk.tokenize import word_tokenize, sent_tokenize  # type: ignore
from bs4 import BeautifulSoup
from langdetect import detect  # type: ignore
import unicodedata
from typing import List


class Preprocessor:
    def remove_urls(self, text: str) -> str:
        """Removes URLs from a given string."""
        url_pattern = r"https?://\S+|www\.\S+"
        return re.sub(url_pattern, "", text)

    def remove_html(self, text: str) -> str:
        """Remove HTML tags."""
        return BeautifulSoup(text, "html.parser").get_text()

    def remove_emojis(self, text: str) -> str:
        """Remove emojis from text."""
        return emoji.replace_emoji(text, "")

    def remove_social_media(self, text: str) -> str:
        """Remove social media handles and hashtags."""
        text = re.sub(r"@\w+", "", text)  # Remove @mentions
        text = re.sub(r"#\w+", "", text)  # Remove hashtags
        return text

    def normalize_text(self, text: str) -> str:
        """Normalize text by replacing accented characters with unaccented ones."""
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )

    def clean_text(self, text: str) -> str:
        """Clean text by removing special characters but keeping letters, numbers, and basic punctuation."""
        # Trim input
        text = text.strip()
        
        # First, normalize multiple punctuation marks
        text = re.sub(r'[!?]+', '', text)  # Remove exclamation and question marks
        
        # Handle ellipsis specially (before general cleaning)
        text = re.sub(r'\.{3,}', '...', text)  # Normalize ellipsis to exactly three dots
        
        # Remove special characters except for basic punctuation
        text = re.sub(r'[^\w\s.,-]', '', text)
        
        # Normalize all whitespace first
        text = re.sub(r'\s+', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s*,\s*', ', ', text)
        text = re.sub(r'\s*\.\s*(?!\.)', '. ', text)
        text = re.sub(r'\s*\.\.\.\s*', '... ', text)
        text = re.sub(r'\s+\.(\s+|$)', '.', text)
        
        # Final cleanup and normalize spaces
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        return text

    def remove_short_lines(self, text: str, min_words: int = 4) -> str:
        """Remove lines with fewer than min_words words."""
        lines = text.split("\n")
        filtered_lines = [line.strip() for line in lines if len(line.split()) > min_words]
        return "\n".join(filtered_lines)

    def remove_duplicate_lines(self, text: str) -> str:
        """Remove duplicate lines while preserving order."""
        seen = set()
        lines = text.split("\n")
        unique_lines = []
        for line in lines:
            line_clean = line.strip()
            if line_clean and line_clean not in seen:
                seen.add(line_clean)
                unique_lines.append(line)
        return "\n".join(unique_lines)

    def remove_repeated_titles(self, text: str) -> str:
        """Remove repeated titles that often appear in news articles."""
        lines = text.split("\n")
        if len(lines) < 2:
            return text
            
        # Check if first two non-empty lines are very similar
        first_lines = [line.strip().lower() for line in lines if line.strip()][:2]
        if len(first_lines) == 2:
            similarity = self._calculate_similarity(first_lines[0], first_lines[1])
            if similarity > 0.8:
                lines = [line for i, line in enumerate(lines) if i != 1]
        
        return "\n".join(lines)

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity ratio between two strings."""
        set1 = set(str1.split())
        set2 = set(str2.split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if union else 0.0

    def clean_quotes(self, text: str) -> str:
        """Standardize quotes and fix common quote-related issues."""
        # Handle various quote types including Portuguese ones
        text = re.sub(r'["""]', '"', text)  # Double quotes
        text = re.sub(r'[''`´]', "'", text)  # Single quotes and accents
        text = re.sub(r'[«»]', '"', text)    # Guillemets (common in Portuguese)
        text = re.sub(r'[‹›]', "'", text)    # Single guillemets
        
        # Fix spacing around quotes
        text = re.sub(r'\s+"', ' "', text)   # Space before opening quote
        text = re.sub(r'"\s+', '" ', text)   # Space after closing quote
        text = re.sub(r'\s+\'', ' \'', text) # Space before single quote
        text = re.sub(r'\'\s+', '\' ', text) # Space after single quote
        
        return text

    def remove_bullet_points(self, text: str) -> str:
        """Remove bullet points and dashes at start of lines."""
        lines = text.split("\n")
        cleaned_lines = [re.sub(r'^[-•*]\s*', '', line.strip()) for line in lines]
        return "\n".join(cleaned_lines)

    def merge_paragraphs(self, text: str) -> str:
        """Merge paragraphs that were split across lines."""
        paragraphs = text.split('\n\n')
        merged_paragraphs = []
        
        for paragraph in paragraphs:
            lines = paragraph.split('\n')
            current_sentence = []
            merged_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # If the previous line doesn't end with sentence-ending punctuation,
                # or this line starts with lowercase, it's a continuation
                if (current_sentence and 
                    not current_sentence[-1].strip().endswith(('.', '!', '?')) or
                    (line and line[0].islower())):
                    current_sentence.append(line)
                else:
                    if current_sentence:
                        merged_lines.append(' '.join(current_sentence))
                    current_sentence = [line]
            
            if current_sentence:
                merged_lines.append(' '.join(current_sentence))
            
            merged_paragraphs.append(' '.join(merged_lines))
        
        return '\n\n'.join(p for p in merged_paragraphs if p.strip())

    def preprocess_text(
        self,
        text: str,
        remove_urls: bool = True,
        remove_emojis: bool = True,
        remove_social: bool = True,
        normalize_unicode: bool = True,
        min_words_per_line: int = 4,
        remove_duplicate_lines: bool = True,
        remove_short_lines: bool = True,
        merge_paragraphs: bool = True,
        clean_text: bool = True,
    ) -> str:
        """Full preprocessing pipeline with configurable steps.

        Args:
            text: Input text to preprocess
            remove_urls: Whether to remove URLs
            remove_emojis: Whether to remove emojis
            remove_social: Whether to remove social media handles/hashtags
            normalize_unicode: Whether to normalize unicode characters
            min_words_per_line: Minimum words required per line
            remove_duplicate_lines: Whether to remove duplicate lines
            remove_short_lines: Whether to remove short lines
            merge_paragraphs: Whether to merge split paragraphs
            clean_text: Whether to perform text cleaning
        """
        if not text.strip():
            return ""

        try:
            lang = detect(text)
        except Exception:
            lang = "en"

        # Initial cleaning
        text = self.remove_bullet_points(text)
        text = self.remove_repeated_titles(text)
        text = re.sub(r"\n+", "\n", text)  # Normalize line breaks

        if remove_urls:
            text = self.remove_urls(text)

        text = self.remove_html(text)

        if remove_emojis:
            text = self.remove_emojis(text)

        if remove_social:
            text = self.remove_social_media(text)

        if normalize_unicode:
            text = self.normalize_text(text)

        # Advanced cleaning
        text = self.clean_quotes(text)
        
        if remove_duplicate_lines:
            text = self.remove_duplicate_lines(text)
            
        if remove_short_lines:
            text = self.remove_short_lines(text, min_words_per_line)
            
        if merge_paragraphs:
            text = self.merge_paragraphs(text)
            
        if clean_text:
            text = self.clean_text(text)

        if not text.strip():
            return ""

        # Tokenize by sentence first, then words
        sentences = sent_tokenize(
            text, language="portuguese" if lang == "pt" else "english"
        )
        processed_sentences = []
        for sentence in sentences:
            tokens = word_tokenize(
                sentence, language="portuguese" if lang == "pt" else "english"
            )
            processed_sentences.append(" ".join(tokens))
        return " ".join(processed_sentences)

