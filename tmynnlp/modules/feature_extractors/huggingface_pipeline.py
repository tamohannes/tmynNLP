from cores import FeatureExtractor
from typing import List, Dict, Union, Any
from transformers import pipeline
from common import util


@FeatureExtractor.register("huggingface_pipeline")
class HuggingFacePipeline(FeatureExtractor):

    def __init__(self,
                 pretrained_model: str = "bert-base-cased",
                 pipeline_name: str = "zero-shot-classification",
                 batch_size: int = 8) -> None:

        self._pipeline = pipeline(pipeline_name, model=pretrained_model)

        self.pretrained_model = pretrained_model
        self.pipeline_name = pipeline_name
        self.batch_size = batch_size

    @FeatureExtractor.cache()
    def __call__(self, input: Union[List[str], str], **kwargs) -> Dict[str, Any]:

        if "labels" in kwargs and isinstance(kwargs["labels"], List):
            labels: List[str] = kwargs["labels"]
        else:
            raise TypeError(f"kwargs['labels'] must be of type: list, found {kwargs['labels'].__class__.__name__} ")

        ret: Dict[str, Any] = []

        if isinstance(input, List):
            batch_group_generator_tqdm = util.get_batch_group_generator_tqdm(
                input, self.batch_size)

            for batch_group in batch_group_generator_tqdm:
                batch_group_output = self._atomic(batch_group, labels)
                ret += batch_group_output
        else:
            ret.append(self._atomic(input))

        return ret

    def _atomic(self, input: str, labels: List[str]) -> List[Dict[str, str]]:
        return self._pipeline(input, labels)
