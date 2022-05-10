from typing import List, Union
from transformers import AutoTokenizer, PreTrainedTokenizer, BatchEncoding
from cores import Tokenizer


@Tokenizer.register("huggingface_tokenizer")
class HuggingFaceTokenizer(Tokenizer):

    def __init__(self,
                 pretrained_model: str = "bert-base-uncased",
                 padding: Union[bool, str] = True,
                 truncation: bool = True,
                 return_tensors: str = "pt",
                 use_fast: bool = True) -> None:

        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            pretrained_model, use_fast=use_fast)

        self.pretrained_model = pretrained_model
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        self.use_fast = use_fast

    @Tokenizer.cache()
    def __call__(self, input: Union[str, List[str]]) -> BatchEncoding:
        return self._tokenizer(input, padding=self.padding, truncation=self.truncation, return_tensors=self.return_tensors)
