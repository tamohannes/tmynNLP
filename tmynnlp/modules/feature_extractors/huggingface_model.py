from cores import FeatureExtractor
from typing import Any, List, Dict
from transformers import AutoModel, PreTrainedModel
from tqdm import tqdm
import torch
from pathlib import Path
from collections import OrderedDict
from common import Device, util
import pickle
import os


@FeatureExtractor.register("huggingface_model")
class HuggingFaceModel(FeatureExtractor):

    def __init__(self,
                 pretrained_model: str = "bert-base-uncased",
                 batch_size: int = 16) -> None:

        self._model: PreTrainedModel = AutoModel.from_pretrained(
            pretrained_model).to(Device.device)

        self.pretrained_model = pretrained_model
        self.batch_size = batch_size

    @FeatureExtractor.cache()
    def __call__(self, input: Any, **kwargs) -> Any:
        def __custom_reduce__(self):
            return (self.__class__, (self.last_hidden_state, self.pooler_output, ))

        with torch.no_grad():
            ret = self._model(**input)

        ret.__reduce__ = __custom_reduce__.__get__(ret)

        return ret

        # # OR - with batching and saving each batch at tmp_dir

        # input_dict: Dict[str, torch.tensor] = input.data
        # seqs_count: int = input.input_ids.shape[0]
        # seq_length: int = input.input_ids.shape[1]

        # # approximate num
        # if seq_length < 250:
        #     batch_group_outputs: List[Any] = []
        #     batch: Dict[str, torch.tensor] = dict()

        #     for batch_start in tqdm(range(0, seqs_count, self.batch_size)):
        #         batch_end = min(batch_start + self.batch_size, seqs_count)

        #         for key in input_dict.keys():
        #             batch[key] = input_dict[key][batch_start: batch_end]

        #         batch_group_outputs.append(self._model(**batch))

        #     batch_group_keys: List[str] = list(batch_group_outputs[0].keys())
        #     batch_group_outputs_concat = OrderedDict()
        #     for key in batch_group_keys:
        #         batch_concat: List[torch.tensor] = []
        #         for batch_group_output in batch_group_outputs:
        #             batch_concat.append(batch_group_output[key].detach())

        #         batch_group_outputs_concat[key] = torch.cat(
        #             tuple(batch_concat), 0)
        # else:
        #     batch_group_outputs: List[Path] = []
        #     batch: Dict[str, torch.tensor] = dict()
        #     batch_group_keys: List[str] = []

        #     for batch_start in tqdm(range(0, seqs_count, self.batch_size)):
        #         file_name_hash: str = Path(util.hash(str(self).split(
        #             ' ')[0][1:])).joinpath(util.hash(str(batch_start)))
        #         file_name_path: Path = self._parent.tmp_dir.joinpath(
        #             file_name_hash)

        #         if not file_name_path.is_file():
        #             batch_end = min(batch_start + self.batch_size, seqs_count)
        #             for key in input_dict.keys():
        #                 batch[key] = input_dict[key][batch_start: batch_end]

        #             with torch.no_grad():
        #                 batch_group_output = self._model(**batch)

        #             os.makedirs(os.path.dirname(file_name_path), exist_ok=True)
        #             with open(file_name_path, "wb") as f:
        #                 batch_group_output.__reduce__ = __custom_reduce__.__get__(
        #                     batch_group_output)
        #                 pickle.dump(batch_group_output, f,
        #                             protocol=pickle.HIGHEST_PROTOCOL)
        #                 del batch_group_output

        #         batch_group_outputs.append(file_name_path)

        #     with open(batch_group_outputs[0], "rb") as f:
        #         batch_sample = pickle.load(f)
        #         batch_group_keys = list(batch_sample.keys())

        #     batch_group_outputs_concat = OrderedDict()
        #     for key in batch_group_keys:
        #         batch_concat: List[torch.tensor] = []
        #         for batch_group_output_name in batch_group_outputs:
        #             with open(batch_group_output_name, "rb") as f:
        #                 batch_group_output = pickle.load(f)
        #             batch_concat.append(batch_group_output[key].detach())

        #         batch_group_outputs_concat[key] = torch.cat(
        #             tuple(batch_concat), 0)

        # ret: Any = batch_group_outputs[0]
        # ret.last_hidden_state = batch_group_outputs_concat['last_hidden_state']
        # ret.pooler_output = batch_group_outputs_concat['pooler_output']
        # # ret = type(rets[0])(last_hidden_state=batch_group_outputs_concat['last_hidden_state'], pooler_output = batch_group_outputs_concat['pooler_output'])

        # ret.__reduce__ = __custom_reduce__.__get__(ret)

        # return ret
