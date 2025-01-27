from setuptools import setup, find_packages

setup(
    name="entity-framing",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "tqdm",
        "nltk",
        "beautifulsoup4",
        "emoji",
        "langdetect"
    ],
) 