import pickle
from typing import Union, Dict, overload, Optional
from keras.preprocessing import sequence


class LanguageIndex():
    def __init__(self, name: str, vocab: list[list[str]]) -> None:
        self.name = name
        self.phrases = vocab
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.vocab = self.__phrases_to_vocab()
        self.max_timesteps = self.__max_timesteps()

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

    def tensor(self, pad: Optional[bool] = True, shift: str = None) -> list[list[int]]:
        tensors = [self[s] for s in self.phrases]
        if shift:
            tensors = self.__shift(tensors, direction=shift)
        if pad:
            return sequence.pad_sequences(
                tensors,
                maxlen=self.max_timesteps,
                padding='post',
                value=self.eos_token,
                truncating='post'
            )
        return tensors

    def save(self, path: str) -> None:
        with open(f"{path}/{self.name}.lang.idx", "wb") as f:
            pickle.dump(self, f)

    def __shift(self, sequences, direction='start') -> list[list[int]]:
        """
        Shift sequences left or right, adding start or end tokens as necessary.

        Args:
        - sequences: List of sequences to be shifted.
        - direction: 'start' to add a start token at the beginning (for decoder inputs),
                     'end' to remove the start token (for targets).
        """
        if direction == 'start':
            return [[self.eos_token] + seq for seq in sequences]
        if direction == 'end':
            return [seq[1:] for seq in sequences]
        raise ValueError("Invalid direction specified. Use 'start' or 'end'.")

    def to_padded_tensor(self, phrases: list[list[str]]) -> list[list[int]]:
        return sequence.pad_sequences(
            [self[s] for s in phrases],
            maxlen=self.max_timesteps,
            padding='post',
            value=self.eos_token
        )

    @property
    def eos_token(self) -> int:
        return 0

    @property
    def zero_word(self) -> str:
        return ''

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + 1

    def __create_index(self) -> None:
        self.idx2word = {self.eos_token: self.zero_word, **dict((idx + 1, word) for idx, word in enumerate(self.vocab))}
        self.word2idx = dict((word, idx) for idx, word in self.idx2word.items())

    def __max_timesteps(self) -> int:
        return max(len(t) for t in self.phrases)

    def __phrases_to_vocab(self) -> list[str]:
        return sorted({word for phrase in self.phrases for word in phrase})

    def __str__(self) -> str:
        vocab_sample = ', '.join(repr(word) for word in list(self.vocab)[:5]) + ('...' if len(self.vocab) > 5 else '')
        return (
            f"LanguageIndex {{ "
            f"sequences: {len(self.phrases)}, "
            f"vocab_size: {self.vocab_size}, "
            f"max_sequence_timestep: {self.max_timesteps}, "
            f"vocab: ({vocab_sample}) }}"
        )
