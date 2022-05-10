from cores import Preprocessor
from datasets import Dataset
import re


@Preprocessor.register("default")
class DefaultPreprocessor(Preprocessor):

    def __init__(self) -> None:
        pass

    @Preprocessor.cache()
    def __call__(self, dataset: Dataset) -> Dataset:

        def dropempty(sample):
            for key in sample.keys():
                if sample[key] == '':
                    return False
            return True

        dataset = dataset.filter(dropempty)
        dataset = dataset.map(lambda sample: {'body': re.sub(
            '\s+\S*$', '', re.sub('^ ', '', re.sub(' +', ' ', re.sub('(.*?(\\xa0)|(\\n).*?)', '', sample['body']))))})

        return dataset
