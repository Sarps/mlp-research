from typing import Union, Dict, overload, Optional

import numpy as np
from keras.preprocessing import sequence
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing


class LanguageIndex():
    def __init__(self, name: str, vocab_path: str, data_path: str, max_timesteps: int) -> None:
        self.name = name
        self.data_path = data_path
        self.tokenizer = self.__setup_tokenizers(vocab_path, max_timesteps)

    @overload
    def __getitem__(self, value: str) -> list[int]:
        ...

    @overload
    def __getitem__(self, value: list[int]) -> str:
        ...

    def __getitem__(self, tokens: Union[list[int], str]) -> Union[int, str, list]:
        if isinstance(tokens, str):
            return self.tokenizer.encode(tokens).ids
        elif isinstance(tokens, list):
            return self.tokenizer.decode(tokens)
        else:
            raise ValueError("value must be 'list' or 'str'")

    def data(self, batch_size=32):
        batch_inputs = []
        while True:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    batch_inputs.append(self[line.strip()])
                    if len(batch_inputs) == batch_size:
                        yield np.array(batch_inputs, ndmin=2)
                        batch_inputs = []

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    def __setup_tokenizers(self, vocab_path: str, max_timesteps: int):
        tokenizer = Tokenizer.from_file(vocab_path)
        tokenizer.enable_padding(length=max_timesteps)
        tokenizer.enable_truncation(max_length=max_timesteps)
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
        )
        return tokenizer

    def __str__(self) -> str:
        return (
            f"LanguageIndex {{ "
            f"name: {self.name}, "
            f"tokenizer: {self.tokenizer}, "
            f"vocab_size: {self.vocab_size} }}"
        )
