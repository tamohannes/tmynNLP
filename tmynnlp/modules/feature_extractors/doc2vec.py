from cores import FeatureExtractor
from typing import Any, Tuple, Union
import numpy as np
from tqdm import tqdm
from gensim.models import Doc2Vec


@FeatureExtractor.register("doc2vec_extract")
class Doc2VecExtract(FeatureExtractor):

    def __init__(self,
                 dm: int = 0,
                 vector_size: int = 300,
                 negative: int = 5,
                 hs: int = 0,
                 min_count: int = 2,
                 sample: int = 0,
                 epochs: int = 20) -> None:

        self.dm = dm
        self.vector_size = vector_size
        self.negative = negative
        self.hs = hs
        self.min_count = min_count
        self.sample = sample
        self.epochs = epochs

    @FeatureExtractor.cache()
    def __call__(self, input: Union[str, Any], **kwargs) -> Tuple[Tuple, Tuple[np.array]]:

        model = Doc2Vec(dm=self.dm, vector_size=self.vector_size,
                        negative=self.negative, hs=self.hs,
                        min_count=self.min_count, sample=self.sample,
                        workers=self._parent.num_workers)
        model.build_vocab([x for x in tqdm(input.values)])

        sents = input.values
        targets, regressors = zip(
            *[(doc.tags[0], model.infer_vector(doc.words, epochs=self.epochs)) for doc in sents])

        return targets, regressors
