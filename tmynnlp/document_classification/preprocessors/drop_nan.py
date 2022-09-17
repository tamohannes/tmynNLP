from typing import Union, Dict, List, Any
from cores import Preprocessor


@Preprocessor.register("drop_nan")
class DropNanPreprocessor(Preprocessor):

    def __init__(self) -> None:
        pass

    def __call__(self, examples: Union[Dict[Any, Any], Dict[List, List]]) -> Dict[Any, Union[Any, List]]:
        keys_to_drop: List[int] = []
        for key in examples.keys():
            if isinstance(examples[key], list):
                for i, sample in enumerate(examples[key]):
                    if sample == "":
                        keys_to_drop.append(i)
            else:
                if examples[key] == "":
                    return {"input": None, "labels": None}

        for key_to_drop in keys_to_drop:
            for key in examples.keys():
                examples[key].pop(key_to_drop)

        return examples
