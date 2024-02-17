# This class creates a word -> index mapping (e.g,. "dad" -> 5) and vice-versa
# (e.g., 5 -> "dad") for each language,
from typing import List, Set


class LanguageIndex():
    def __init__(self, vocab: Set[str]) -> None:
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = sorted(vocab)

        self.create_index()

    def create_index(self) -> None:
        self.idx2word = {0: '', **dict((idx + 1, word) for idx, word in enumerate(self.vocab))}
        self.word2idx = dict((word, idx) for idx, word in self.idx2word.items())
