from typing import Union, Dict, overload, Optional
from keras.preprocessing import sequence


class LanguageIndex():
    def __init__(self, vocab: list[list[str]]) -> None:
        self.phrases = vocab
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.vocab = self.__phrases_to_vocab()
        self.max_length = self.__max_length()

        self.__create_index()

    @overload
    def __getitem__(self, value: str) -> int:
        ...

    @overload
    def __getitem__(self, value: int) -> str:
        ...

    @overload
    def __getitem__(self, value: list[str]) -> list[int]:
        ...

    @overload
    def __getitem__(self, value: list[int]) -> list[str]:
        ...

    def __getitem__(self, key: Union[int, str, list]) -> Union[int, str, list]:
        if isinstance(key, str):
            return self.word2idx[key]
        elif isinstance(key, int):
            return self.idx2word[key]
        elif isinstance(key, list):
            return [self[token] for token in key]
        else:
            raise ValueError("value must be 'int' or 'str'")

    def tensor(self, pad: Optional[bool] = True) -> list[list[int]]:
        tensors = [self[s] for s in self.phrases]
        if pad:
            return sequence.pad_sequences(tensors, maxlen=self.max_length, padding='post', value=self.zero_idx)
        return tensors

    def to_padded_tensor(self, phrases: list[list[str]]) -> list[int]:
        return sequence.pad_sequences(
            [self[s] for s in phrases],
            maxlen=self.max_length,
            padding='post',
            value=self.zero_idx
        )

    @property
    def zero_idx(self) -> int:
        return 0

    @property
    def zero_word(self) -> str:
        return ''

    def __create_index(self) -> None:
        self.idx2word = {self.zero_idx: self.zero_word, **dict((idx + 1, word) for idx, word in enumerate(self.vocab))}
        self.word2idx = dict((word, idx) for idx, word in self.idx2word.items())

    def __max_length(self) -> int:
        return max(len(t) for t in self.phrases)

    def __phrases_to_vocab(self) -> list[str]:
        return sorted({word for phrase in self.phrases for word in phrase})

    def __str__(self) -> str:
        vocab_sample = ', '.join(repr(word) for word in list(self.vocab)[:5]) + ('...' if len(self.vocab) > 5 else '')
        return (
            f"LanguageIndex {{ "
            f"sequences: {len(self.phrases)}, "
            f"vocab_size: {len(self.vocab)}, "
            f"max_sequence_timestep: {self.max_length}, "
            f"vocab: ({vocab_sample}) }}"
        )
