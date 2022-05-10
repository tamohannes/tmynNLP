import importlib
import os
import sys
from pathlib import Path
import pkgutil
from typing import Generator, TypeVar, List, Any, Union, Iterable, Iterator
from itertools import islice
from tqdm import tqdm
import math
import hashlib


PathType = Union[os.PathLike, str]
T = TypeVar("T")
ContextManagerFunctionReturnType = Generator[T, None, None]


def push_python_path(path: PathType) -> ContextManagerFunctionReturnType[None]:
    """
    Prepends the given path to `sys.path`.

    This method is intended to use with `with`, so after its usage, its value willbe removed from
    `sys.path`.
    """
    # In some environments, such as TC, it fails when sys.path contains a relative path, such as ".".
    path = Path(path).resolve()
    path = str(path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        # Better to remove by value, in case `sys.path` was manipulated in between.
        sys.path.remove(path)


def import_module_and_submodules(package_name: str) -> None:
    """
    Import all submodules under the given package.
    Primarily useful so that people using AllenNLP as a library
    can specify their own custom packages and have their custom
    classes get loaded and registered.
    """
    importlib.invalidate_caches()

    # For some reason, python doesn't always add this by default to your path, but you pretty much
    # always want it when using `--include-package`.  And if it's already there, adding it again at
    # the end won't hurt anything.
    # with push_python_path("."):
    # Import at top level
    module = importlib.import_module(package_name)
    path = getattr(module, "__path__", [])
    path_string = "" if not path else path[0]

    # walk_packages only finds immediate children, so need to recurse.
    for module_finder, name, _ in pkgutil.walk_packages(path):
        # Sometimes when you import third-party libraries that are on your path,
        # `pkgutil.walk_packages` returns those too, so we need to skip them.
        if path_string and module_finder.path != path_string:
            continue
        subpackage = f"{package_name}.{name}"
        import_module_and_submodules(subpackage)


A = TypeVar("A")


def lazy_groups_of(iterable: Iterable[A], group_size: int) -> Iterator[List[A]]:
    """
    Takes an iterable and batches the individual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    """
    iterator = iter(iterable)
    while True:
        s = list(islice(iterator, group_size))
        if len(s) > 0:
            yield s
        else:
            break


def get_batch_group_generator_tqdm(input: List[Any], batch_size: int) -> tqdm:
    """
    Takes a list of any elements and returns a generator wrapped with tqdm
    """
    batch_generator = iter(input)
    batch_group_generator = lazy_groups_of(batch_generator, batch_size)

    num_total_steps: int = math.ceil(len(input) / batch_size)
    batch_group_generator_tqdm = tqdm(
        batch_group_generator, total=num_total_steps)

    return batch_group_generator_tqdm


def hash(name: str, hash_method: str = "short") -> str:
    """
    Takes a string with arbitrary length and returns the hash of it encoded in utf-8
    """
    if hash_method == "long":
        return hashlib.sha224(bytes(name, encoding='utf-8')).hexdigest()
    else:
        return hashlib.md5(bytes(name, encoding='utf-8')).hexdigest()
