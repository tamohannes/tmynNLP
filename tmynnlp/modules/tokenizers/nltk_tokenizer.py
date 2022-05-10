from typing import List, Union
from cores import Tokenizer
import nltk
from common import util


@Tokenizer.register("nltk_tokenizer")
class NLTKTokenizer(Tokenizer):

    def __init__(self, batch_size: int = 8) -> None:

        self.batch_size = batch_size

    @Tokenizer.cache()
    def __call__(self, input: Union[str, List[str]]) -> List[List[str]]:
        ret: List[List[str]] = []

        if isinstance(input, List):
            batch_group_generator_tqdm = util.get_batch_group_generator_tqdm(
                input, self.batch_size)

            for batch_group in batch_group_generator_tqdm:
                for batch in batch_group:
                    ret.append(self._atomic(batch))
        else:
            ret.append(self._atomic(input))

        return ret

    def _atomic(self, input: str) -> List[str]:
        tokens: List[str] = []

        for word in nltk.word_tokenize(input):
            if not word:
                continue
            tokens.append(word.lower())

        return tokens
