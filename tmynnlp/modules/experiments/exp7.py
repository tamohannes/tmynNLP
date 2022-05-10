from pathlib import Path
from typing import List, Tuple
from cores import DatasetReader, Experiment, Metric, Preprocessor, Tokenizer, FeatureExtractor
from common import Device, TmpHandler
import torch
from datasets import ClassLabel
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer


@Experiment.register("exp7")
class Exp7(Experiment):

    def __init__(self,
                 dataset_reader: DatasetReader,
                 preprocessor: Preprocessor,
                 metrics: List[Metric],
                 tokenizer: Tokenizer,
                 feature_extractor: FeatureExtractor = None,
                 num_workers: int = -1,
                 classifier_pipeline: str = "bert-base-cased",
                 batch_size: int = 16,
                 seed: int = 17,
                 evaluation_strategy: str = "epoch",
                 num_train_epochs: int = 3,
                 learning_rate: float = 5e-04) -> None:

        super().__init__(dataset_reader, preprocessor, metrics, tokenizer,
                         feature_extractor, num_workers,
                         classifier_pipeline=classifier_pipeline,
                         batch_size=batch_size, seed=seed,
                         evaluation_strategy=evaluation_strategy,
                         num_train_epochs=num_train_epochs,
                         learning_rate=learning_rate)

    def __call__(self) -> Tuple[List[str], List[str]]:

        self.datasets['train'] = self.preprocessor(self.datasets['train'])
        self.datasets['val'] = self.preprocessor(self.datasets['val'])

        """#Base Case"""

        labels_raw = set.union(set(self.datasets['train']['matter']), set(
            self.datasets['val']['matter']))
        labels = ClassLabel(num_classes=len(
            labels_raw), names=list(labels_raw))

        self.datasets['train'].rename_column_('matter', 'label')
        self.datasets['train'].rename_column_('body', 'text')
        self.datasets['train'] = self.datasets['train'].remove_columns('subject')

        self.datasets['val'].rename_column_('matter', 'label')
        self.datasets['val'].rename_column_('body', 'text')
        self.datasets['val'] = self.datasets['val'].remove_columns('subject')

        def tokenize_function(batch):
            tokenized_batch = self.tokenizer(batch['text'])
            tokenized_batch['input_ids'] = tokenized_batch['input_ids'].numpy()
            tokenized_batch['token_type_ids'] = tokenized_batch['token_type_ids'].numpy(
            )
            tokenized_batch['attention_mask'] = tokenized_batch['attention_mask'].numpy(
            )
            tokenized_batch['label'] = labels.str2int(batch['label'])

            return tokenized_batch

        self.datasets['train'] = self.datasets['train'].map(
            tokenize_function, batched=True, batch_size=self.batch_size).shuffle(seed=self.seed)
        self.datasets['val'] = self.datasets['val'].map(
            tokenize_function, batched=True, batch_size=self.batch_size).shuffle(seed=self.seed)

        self.datasets['train'].set_format(
            'torch', columns=['input_ids', 'attention_mask', 'label'])
        self.datasets['val'].set_format(
            'torch', columns=['input_ids', 'attention_mask', 'label'])

        output_dir: Path = TmpHandler.get_path(
            self, self.classifier_pipeline)

        training_args = TrainingArguments(
            output_dir, evaluation_strategy=self.evaluation_strategy, num_train_epochs=self.num_train_epochs)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.classifier_pipeline, num_labels=len(labels_raw)).to(Device.device)

        # Fixed Feature Extraction
        for param in model.base_model.parameters():
            param.requires_grad = False

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.datasets['train'],
            eval_dataset=self.datasets['val']
        )

        trainer.train()

        _, preds = torch.max(torch.tensor(trainer.predict(
            self.datasets['val']).predictions), axis=1)
        predictions = preds.tolist()
        gold_labels = [int(self.datasets['val'][i]['label'])
                       for i in range(len(self.datasets['val']))]

        return gold_labels, predictions

    def info(self) -> str:
        return "Transfer-Learning: Apply Fixed-Feature Extraction technique using the labeled data."
