from cores import FeatureExtractor, Tokenizer
from typing import List, Dict, Union
from transformers import AutoModelForTokenClassification
from transformers import pipeline
from common import Device


@FeatureExtractor.register("huggingface_ner")
class HuggingFaceNer(FeatureExtractor):

    def __init__(self,
                 ner_tokenizer: Tokenizer,
                 pretrained_model: str = "dslim/bert-base-NER",
                 ner_classifier: str = "ner",
                 ner_keys: List[str] = ["B-MIS", "I-MIS", "B-PER",
                                        "I-PER", "B-ORG", "I-ORG",
                                        "B-LOC", "I-LOC"],
                 batch_size: int = 8) -> None:

        ner_model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model).to(Device.device)
        self._ner_extractor = pipeline(
            ner_classifier, model=ner_model, tokenizer=ner_tokenizer._tokenizer)

        self.ner_tokenizer = ner_tokenizer
        self.pretrained_model = pretrained_model
        self.ner_classifier = ner_classifier
        self.ner_keys = ner_keys
        self.batch_size = batch_size

    @FeatureExtractor.cache()
    def __call__(self, input: Union[List[str], str], **kwargs) -> List[List[Dict[str, str]]]:
        ret: List[List[Dict[str, str]]] = []

        if isinstance(input, List):
            for batch_group in input:
                batch_ner = self._atomic(batch_group)
                ret.append(batch_ner)
        else:
            ret.append(self._atomic(input))

        return ret

    def _atomic(self, input: str) -> List[Dict[str, str]]:
        sample_ner = self._ner_extractor(input)
        batch_ners_filtered = [
            ner for ner in sample_ner if ner['entity'] in self.ner_keys]
        return batch_ners_filtered
