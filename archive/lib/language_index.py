from typing import Union, overload

from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from .loaders import read_lines


class LanguageIndex():
    def __init__(self, name: str, vocab_path: str, max_timesteps: int, data_path: str = None) -> None:
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

    def data(self, training_sample: int = None):
        if self.data_path is None:
            raise AttributeError("No data path specified")
        return [self[line] for line in read_lines(self.data_path, training_sample)]

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    @property
    def sos_token_id(self) -> int:
        return 0

    @property
    def eos_token_id(self) -> int:
        return 1

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
