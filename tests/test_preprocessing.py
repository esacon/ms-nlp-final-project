import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import Preprocessor
import pytest


@pytest.fixture
def preprocessor():
    """Fixture to initialize the Preprocessor instance."""
    return Preprocessor()


def test_remove_urls(preprocessor):
    text = "Check this link: https://example.com and this one: www.example.org"
    cleaned_text = preprocessor.remove_urls(text)
    assert cleaned_text == "Check this link:  and this one: "


def test_remove_html(preprocessor):
    text = "<p>This is a <b>test</b>.</p>"
    cleaned_text = preprocessor.remove_html(text)
    assert cleaned_text == "This is a test."


def test_remove_emojis(preprocessor):
    text = "I love Python 😊🐍❤️!"
    cleaned_text = preprocessor.remove_emojis(text)
    assert cleaned_text == "I love Python !"


def test_remove_social_media(preprocessor):
    text = "Hello @user! Check out #Python and follow @OpenAI."
    cleaned_text = preprocessor.remove_social_media(text)
    assert cleaned_text == "Hello ! Check out  and follow ."


def test_normalize_text(preprocessor):
    text = "Café déjà vu résumé"
    normalized_text = preprocessor.normalize_text(text)
    assert normalized_text == "Cafe deja vu resume"


def test_clean_text(preprocessor):
    text = "Hello!! This text has... special characters: @#$%^&*()."
    cleaned_text = preprocessor.clean_text(text)
    assert cleaned_text == "Hello This text has... special characters."


def test_remove_short_lines(preprocessor):
    text = "Short\nThis is a longer line\nTiny\nAnother valid line it is"
    filtered_text = preprocessor.remove_short_lines(text)
    assert filtered_text == "This is a longer line\nAnother valid line it is"


def test_remove_duplicate_lines(preprocessor):
    text = "First line\nSecond line\nFirst line\nThird line"
    cleaned_text = preprocessor.remove_duplicate_lines(text)
    assert cleaned_text == "First line\nSecond line\nThird line"


def test_remove_repeated_titles(preprocessor):
    text = "Breaking News: Important Event\nBREAKING NEWS: IMPORTANT EVENT\nDetails follow..."
    cleaned_text = preprocessor.remove_repeated_titles(text)
    assert cleaned_text == "Breaking News: Important Event\nDetails follow..."


def test_clean_quotes(preprocessor):
    text = 'He said "hello" and \'hi\' and "goodbye"'
    cleaned_text = preprocessor.clean_quotes(text)
    assert cleaned_text == 'He said "hello" and \'hi\' and "goodbye"'


def test_remove_bullet_points(preprocessor):
    text = "• First point\n- Second point\n* Third point"
    cleaned_text = preprocessor.remove_bullet_points(text)
    assert cleaned_text == "First point\nSecond point\nThird point"


def test_merge_paragraphs(preprocessor):
    text = "This is a sentence\nthat continues here.\n\nNew paragraph."
    merged_text = preprocessor.merge_paragraphs(text)
    assert merged_text == "This is a sentence that continues here.\n\nNew paragraph."


def test_preprocess_text_english(preprocessor):
    text = """Title
    
    This is a test article. It has multiple sentences.
    
    • First bullet point
    • Second point
    
    "Quote here" and more text."""
    
    processed = preprocessor.preprocess_text(text, min_words_per_line=2)
    # Check key aspects of processing
    assert "bullet" in processed
    assert "Quote" in processed
    assert "•" not in processed
    assert "\n\n" not in processed  # No double newlines


def test_preprocess_text_portuguese(preprocessor):
    text = """Título do Artigo
    
    Este é um texto em português. Tem várias frases.
    
    - Primeiro ponto
    - Segundo ponto
    
    "Citação aqui" e mais texto."""
    
    # Test with unicode normalization
    processed_norm = preprocessor.preprocess_text(text, normalize_unicode=True)
    assert "e" in processed_norm  # Should have normalized é to e
    
    # Test without unicode normalization
    processed_no_norm = preprocessor.preprocess_text(text, normalize_unicode=False)
    assert "é" in processed_no_norm  # Should preserve é


def print_comparison(original: str, processed: str, title: str = "Test Case") -> None:
    """Print a before/after comparison of text processing."""
    print(f"\n{'='*20} {title} {'='*20}")
    print("\nORIGINAL TEXT:")
    print('-' * 60)
    print(original)
    print('\nPROCESSED TEXT:')
    print('-' * 60)
    print(processed)
    print('=' * 60)


def main():
    # Initialize preprocessor
    preprocessor = Preprocessor()

    # Test Case 1: English article with repeated title and quotes
    en_article = '''Oxford Residents Mount Resistance Against the Sectioning of Their Streets 

 OXFORD RESIDENTS MOUNT RESISTANCE AGAINST THE SECTIONING OF THEIR STREETS

THE GREAT CLIMATE CON

Oxford residents are taking matters into their own hands and destroying the street zone sectioning barriers.

"It's very inspiring to see the people of the UK take matters into their own hands," said John Smith. "This year is looking to be jam packed with resistance."

• First point of action
• Second major development
• Third key issue

Further action planned for next week...'''

    # Test Case 2: Portuguese article with accents and bullet points
    pt_article = '''Zequinha critica UE por adiar obrigatoriedade de preservação ambiental

O senador Zequinha Marinho (Podemos-PA) criticou, em pronunciamento na quarta-feira (28), a decisão da União Europeia (UE).

- Primeiro ponto importante
- Segundo aspecto relevante

"Como a hipocrisia é alguma coisa que não podemos compreender com esta turma. Não querem cumprir 4%, mas exigem que os brasileiros reservem 80% para a preservação ambiental."

Para o senador, a decisão do Parlamento Europeu é um mecanismo de protecionismo comercial.'''

    # Process English article
    processed_en = preprocessor.preprocess_text(
        en_article,
        normalize_unicode=True,
        min_words_per_line=4
    )
    print_comparison(en_article, processed_en, "English Article")

    # Process Portuguese article
    # Test with and without unicode normalization
    processed_pt_with_norm = preprocessor.preprocess_text(
        pt_article,
        normalize_unicode=True,
        min_words_per_line=4
    )
    print_comparison(pt_article, processed_pt_with_norm,
                     "Portuguese Article (with normalization)")

    processed_pt_without_norm = preprocessor.preprocess_text(
        pt_article,
        normalize_unicode=False,
        min_words_per_line=4
    )
    print_comparison(pt_article, processed_pt_without_norm,
                     "Portuguese Article (without normalization)")


if __name__ == "__main__":
    main()
