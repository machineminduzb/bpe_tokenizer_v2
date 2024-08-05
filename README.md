BPE Tokenizer
This package provides an implementation of Byte Pair Encoding (BPE) tokenizer, designed to be efficient and fast. It includes training on both CPU and GPU, along with encoding and decoding functionalities.

Features
Train BPE tokenizer on CPU and GPU
Encode and decode text using the trained tokenizer
Save and load merges and vocabulary to/from JSON files
Support for special tokens and customizable regex patterns
Installation
To install the package, use the following command:

bash
Copy code
pip install bpe-tokenizer
Usage
Tokenizer Training on CPU
To train a tokenizer on CPU:

python
Copy code
from bpe_tokenizer import TrainTokenizer

text = "your training text here"
tokenizer = TrainTokenizer(text)
merges = tokenizer.train(vocab_size=1256, merges_file='merges.json', vocab_file='vocab.json')
Tokenizer Training on GPU
To train a tokenizer on GPU:

python
Copy code
from bpe_tokenizer import TrainTokenizerGPU

text = "your training text here"
tokenizer = TrainTokenizerGPU(text)
merges = tokenizer.train(vocab_size=1256, merges_file='merges.json', vocab_file='vocab.json')
Encoding and Decoding Text
To encode and decode text using the trained tokenizer:

python
Copy code
from bpe_tokenizer import Encoder

encoder = Encoder(merges_file_path='merges.json', vocab_file_path='vocab.json', special_tokens=['[PAD]', '[UNK]', '[STR]', '[END]', '[SEP]'])
encoded = encoder.encode("your text here")
decoded = encoder.decode(encoded)

print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
Classes and Methods
TrainTokenizer
A class to train a tokenizer on a CPU device using the BPE algorithm.

Attributes

data (np.ndarray): The input text converted to a numpy array of bytes.
Methods

__init__(self, text: str): Initialize with input text.
_get_stats(self) -> np.ndarray: Find the most frequent pair of adjacent bytes in the data.
_merge(self, ids: np.ndarray, pair: np.ndarray, idx: int) -> np.ndarray: Merge the most frequent pair in the data.
train(self, vocab_size: int = 1256, merges_file: str = 'merges.json', vocab_file: str = 'vocab.json') -> Dict[Tuple[int, int], int]: Train the tokenizer.
_save_merges_json(self, merges: Dict[Tuple[int, int], int], path: str) -> None: Save the merges to a JSON file.
_save_vocab_json(self, merges: Dict[Tuple[int, int], int], path: str) -> None: Save the vocabulary to a JSON file.
TrainTokenizerGPU
A class to train a tokenizer using GPU acceleration with the BPE algorithm.

Attributes

data (np.ndarray): The input text converted to a numpy array of bytes.
Methods

__init__(self, text: str): Initialize with input text.
_get_stats(self, ids: torch.Tensor) -> torch.Tensor: Find the most common pair of adjacent tokens in the data.
_merge(self, ids: torch.Tensor, pair: torch.Tensor, idx: int) -> torch.Tensor: Merge the most frequent pair in the data.
train(self, vocab_size: int = 1256, merges_file: str = 'merges.json', vocab_file: str = 'vocab.json') -> Dict[Tuple[int, int], int]: Train the tokenizer.
_save_merges_json(self, merges: Dict[Tuple[int, int], int], path: str) -> None: Save the merges to a JSON file.
_save_vocab_json(self, merges: Dict[Tuple[int, int], int], path: str) -> None: Save the vocabulary to a JSON file.
Encoder
A class for encoding and decoding text using byte-pair encoding (BPE).

Attributes

regex (re.Pattern): A compiled regex pattern for initial tokenization.
special_tk (List[str]): A list of special tokens.
merges (Dict[Tuple[int, int], int]): The trained merges.
vocab (Dict[int, bytes]): The vocabulary mapping token IDs to byte sequences.
Methods

__init__(self, merges_file_path: str = "merges.json", vocab_file_path: str = "vocab.json", special_tokens: List[str] = ['[PAD]', '[UNK]', '[STR]', '[END]', '[SEP]'], regex_pattern: str = r""" ?\b(?:\w*'\w*)+\b|\[[A-Z]{3}\]| ?\b\w+\b| ?[,.!?(){}["-\]]|\s"""): Initialize the Encoder with special tokens and load the trained model.
encode(self, text: str): Encode the input text into token IDs.
decode(self, ids: List[int]): Decode the input list of integers into a string.
_encode_chunk(self, text: str): Encode a single chunk of text.
_encode_merge(self, ids, pairs, idx): Perform merges on the token IDs.
_load_vocab(self, file_path): Load vocabulary from JSON file.
_load_merges(self, file_path): Load merges from JSON file.
License
This project is licensed under the MIT License.

Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

Contact
For any questions or suggestions, feel free to open an issue on the GitHub repository.