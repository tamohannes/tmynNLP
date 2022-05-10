from .registrable import Registrable
from common import util
from pathlib import Path
from datasets import Dataset
from typing import TypeVar, Type, Dict, List, Tuple, Any, Callable
import logging
import torch
import json
import pickle
import copy
import os


C = TypeVar("C")


class Cacheable:
    cache_dir: Path
    exceptional_properties: List[str] = ["batch_size"]

    @classmethod
    def make_dir(cls, cache_dir: str):
        Cacheable.cache_dir = Path(cache_dir)
        Cacheable.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _pop_extra_properties(cls, properties: Dict[str, Any]):
        for name in list(properties.keys()):
            # protected properties
            if name.startswith("_"):
                properties.pop(name)
            # exceptions
            elif name in Cacheable.exceptional_properties:
                properties.pop(name)

        return properties

    @classmethod
    def _extract_action_features(cls, method: Callable, module: Type[C]) -> Tuple[str, str]:
        action_name: str = str(method).split(".")[-1].split(" ")[0]
        action_path: str = str(module).split(" ")[0][1:]
        arguments: Dict[str, Any] = Cacheable._pop_extra_properties(
            copy.copy(vars(module)))

        action_full_name: str = f"{action_path}.{action_name}"
        arguments_str: str = Cacheable._json_serialize(arguments)

        return action_full_name, arguments_str

    @classmethod
    def _json_serialize(cls, input: Any) -> str:
        try:
            ret: str = json.dumps(input)
            return ret
        except:
            # if is dict like
            if callable(getattr(input, 'keys', None)):
                ret_list: List[str] = []
                for input_key in input.keys():
                    ret_list.append(
                        Cacheable._json_serialize(input[input_key]))
                ret: str = json.dumps(ret_list)
            elif isinstance(input, tuple) or isinstance(input, list):
                ret_list: List[str] = []
                for input_elem in input:
                    ret_list.append(Cacheable._json_serialize(input_elem))
                ret: str = json.dumps(ret_list)
            elif isinstance(input, torch.Tensor):
                ret: str = json.dumps(input.cpu().numpy().tolist())
            elif isinstance(input, Dataset):
                ret: str = json.dumps(input.to_dict())
            else:
                ret: str = Cacheable._json_serialize(vars(input))

            return ret

    @classmethod
    def action_name_for_logging(cls: Type[C], action_name) -> str:
        return "->".join(action_name.lower().split(".")[-2:])

    @classmethod
    def cache(cls: Type[C]):
        def initialized(callback: Callable):
            def called(*args, **kwargs):
                self = args[0]
                arguments = args[1:]

                components: List[Any] = []
                for component in vars(self._parent).values():
                    if isinstance(component, Registrable):
                        if component == self:
                            break
                        components.append(component)

                components_prefix: List[str] = []
                for component in components:
                    component_action_name: str = str(
                        component).split(" ")[0][1:]
                    component_arguments: str = Cacheable._json_serialize(
                        Cacheable._pop_extra_properties(copy.copy(vars(component))))

                    components_prefix.append(
                        f"{component_action_name}.{component_arguments}")

                action_name, action_arguments = Cacheable._extract_action_features(
                    callback, self)

                components_prefix.append(f"{action_name}.{action_arguments}")

                arguments_str: str = Cacheable._json_serialize(arguments)

                cache_file_name: str = "-".join(components_prefix)
                cache_file_name_hash: Path = Path(util.hash(cache_file_name)).joinpath(
                    util.hash(arguments_str, 'long'))
                cache_file_path: Path = Cacheable.cache_dir.joinpath(
                    Path(cache_file_name_hash))

                if cache_file_path.is_file():
                    logging.info(f"using cache for {Cacheable.action_name_for_logging(action_name)} from {cache_file_path}")
                    with open(cache_file_path, "rb") as f:
                        callback_result = pickle.load(f)
                else:
                    logging.info(f"starting {Cacheable.action_name_for_logging(action_name)}")
                    callback_result = callback(*args, **kwargs)
                    logging.info(f"finished {Cacheable.action_name_for_logging(action_name)}")
                    os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
                    with open(cache_file_path, "wb") as f:
                        pickle.dump(callback_result, f, protocol=pickle.HIGHEST_PROTOCOL)

                return callback_result
            return called
        return initialized

    def reverse_registration(instance: Type[C], parent: Any) -> None:
        instance._parent = parent
