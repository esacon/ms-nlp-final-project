import os
import sys
import pytest
import tempfile
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader, Article, EntityAnnotation


@pytest.fixture
def sample_data_dir():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create train data
        for lang in ["EN", "PT"]:
            # Train split
            train_docs_dir = os.path.join(tmp_dir, "train", lang, "raw-documents")
            os.makedirs(train_docs_dir, exist_ok=True)

            article_id = f"{lang}_train_10001.txt"
            with open(
                os.path.join(train_docs_dir, article_id), "w", encoding="utf-8"
            ) as f:
                f.write("Train article content")

            with open(
                os.path.join(tmp_dir, "train", lang, "subtask-1-annotations.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(
                    f"{article_id}\tEntity Name\t0\t10\tProtagonist\tHero\tMartyr\n"
                )

            # Dev split
            dev_docs_dir = os.path.join(tmp_dir, "dev", lang, "raw-documents")
            os.makedirs(dev_docs_dir, exist_ok=True)

            article_id = f"{lang}_dev_10001.txt"
            with open(
                os.path.join(dev_docs_dir, article_id), "w", encoding="utf-8"
            ) as f:
                f.write("Dev article content")

            with open(
                os.path.join(tmp_dir, "dev", lang, "raw-entity-mentions.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(f"{article_id}\tEntity Name\t0\t10\n")

        yield tmp_dir


def test_load_data(sample_data_dir):
    loader = DataLoader(sample_data_dir)
    loader.load_data()

    assert len(loader.articles) == 4

    # Test train article
    train_article = loader.articles["EN_train_10001.txt"]
    assert train_article.content == "Train article content"
    assert train_article.language == "EN"
    assert train_article.split == "train"
    assert len(train_article.annotations) == 1
    assert train_article.annotations[0].main_role == "Protagonist"

    # Test dev article
    dev_article = loader.articles["EN_dev_10001.txt"]
    assert dev_article.content == "Dev article content"
    assert dev_article.language == "EN"
    assert dev_article.split == "dev"
    assert len(dev_article.annotations) == 1
    assert dev_article.annotations[0].main_role is None


def test_get_articles_by_split(sample_data_dir):
    loader = DataLoader(sample_data_dir)
    loader.load_data()

    train_articles = loader.get_articles_by_split("train")
    assert len(train_articles) == 2

    dev_articles = loader.get_articles_by_split("dev")
    assert len(dev_articles) == 2


def test_get_articles_by_language(sample_data_dir):
    loader = DataLoader(sample_data_dir)
    loader.load_data()

    en_articles = loader.get_articles_by_language("EN")
    assert len(en_articles) == 2
    assert all(article.language == "EN" for article in en_articles)


def test_missing_directory():
    loader = DataLoader("/nonexistent/path")
    loader.load_data()
    assert len(loader.articles) == 0
