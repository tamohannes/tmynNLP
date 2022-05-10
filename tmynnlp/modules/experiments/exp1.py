from pathlib import Path
from typing import List, Tuple
from cores import DatasetReader, Experiment, Metric, Preprocessor, Tokenizer, FeatureExtractor
from gensim.models.doc2vec import TaggedDocument
from sklearn.linear_model import LogisticRegression
import pandas as pd


@Experiment.register("exp1")
class Exp1(Experiment):

    def __init__(self,
                 dataset_reader: DatasetReader,
                 preprocessor: Preprocessor,
                 metrics: List[Metric],
                 tokenizer: Tokenizer,
                 feature_extractor: FeatureExtractor,
                 num_workers: int = -1,
                 C: float = 1e5) -> None:

        super().__init__(dataset_reader, preprocessor, metrics, tokenizer,
                         feature_extractor, num_workers, C=C)

    def __call__(self) -> Tuple[List[str], List[str]]:

        train_tok = self.tokenizer(self.datasets['train']['body'])
        train_matter = self.datasets['train']['matter']

        val_tok = self.tokenizer(self.datasets['val']['body'])
        val_matter = self.datasets['val']['matter']

        train_tagged = []
        for i in range(len(train_tok)):
            train_tagged.append(TaggedDocument(
                words=train_tok[i], tags=train_matter[i]))

        train_tagged = pd.Series(train_tagged)

        val_tagged = []
        for i in range(len(val_tok)):
            val_tagged.append(TaggedDocument(
                words=val_tok[i], tags=val_matter[i]))

        val_tagged = pd.Series(val_tagged)

        y_train, X_train = self.feature_extractor(train_tagged)
        gold_labels, X_val = self.feature_extractor(val_tagged)

        logreg = LogisticRegression(n_jobs=self.num_workers, C=self.C)
        logreg.fit(X_train, y_train)

        predictions = logreg.predict(X_val)

        return gold_labels, predictions

    def info(self) -> str:
        return "Doc2Vec: On the last layer Logistic Regression Classifier"
