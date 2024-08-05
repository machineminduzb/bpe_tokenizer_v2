from setuptools import setup, find_packages

setup(
    name="bpe_tokenizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "torch>=1.7.0",
        "tqdm>=4.45.0",
        "regex>=2020.4.4"
    ]
)