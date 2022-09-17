from typing import List, Tuple, Optional, Dict
from cores import DatasetReader, Experiment, Metric, Tokenizer, Model, Tracker, Criterion, Optimizer, LRScheduler
from common import Device, Params
from tqdm.auto import tqdm
import torch


@Experiment.register("experiment1")
class Experiment1(Experiment):

    def __init__(self,
                 dataset_reader: DatasetReader,
                 metrics: List[Metric],
                 tokenizer: Tokenizer,
                 model: Model,
                 tracker: Tracker,
                 run_params: Params,
                 num_epochs: int,
                 criterion: Dict,
                 optimizer: Dict,
                 lr_scheduler: Dict,
                 validate_every_n_epoch: Optional[int] = None,
                 num_workers: int = -1,
                 batch_size: int = 4,) -> None:

        super().__init__(dataset_reader, metrics, tokenizer, model, tracker, run_params, num_workers,
                         batch_size=batch_size, num_epochs=num_epochs, criterion=criterion, optimizer=optimizer,
                         lr_scheduler=lr_scheduler, validate_every_n_epoch=validate_every_n_epoch)


    def __call__(self) -> Tuple[List[str], List[str]]:

        self.datasets['train'] = self.datasets['train'].map(self.tokenizer, batched=True)
        self.datasets['valid'] = self.datasets['valid'].map(self.tokenizer, batched=True)

        self.datasets['train'].set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
        self.datasets['valid'].set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

        train_dataloader = torch.utils.data.DataLoader(self.datasets['train'], batch_size=self.batch_size)
        valid_dataloader = torch.utils.data.DataLoader(self.datasets['valid'], batch_size=self.batch_size)

        criterion = Criterion(self.criterion)
        optimizer = Optimizer(self.model.parameters(), self.optimizer)
        lr_scheduler = LRScheduler(optimizer, self.lr_scheduler)

        for epoch in range(1, self.num_epochs+1):
            self.train(train_dataloader, criterion,optimizer, lr_scheduler, epoch)
            if self.validate_every_n_epoch and epoch % self.validate_every_n_epoch == 0:
                self.validate(valid_dataloader)
        self.validate(valid_dataloader)


    def train(self, dataloader, criterion, optimizer, lr_scheduler, epoch):
        self.model.train()

        train_progress_bar = tqdm(total=int(self.datasets['train'].num_rows/self.batch_size))
        train_progress_bar.set_description(f"training: epoch {epoch}/{self.num_epochs}")

        context = {"subset": "train"} if self.model.training else {"subset": "valid"}

        for batch in dataloader:
            batch = {k: v.to(Device.device) for k, v in batch.items()}
            outputs = self.model(**batch)

            loss = criterion(outputs.logits, batch["labels"])
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_progress_bar.update(1)

            logs = {"loss": loss.item()}
            self.tracker.track(logs, context=context)

        train_progress_bar.set_description(f"training: epoch {epoch}/{self.num_epochs} | loss:{'%.2f' % loss.item()}")
        train_progress_bar.close()


    def validate(self, dataloader):
        self.model.eval()

        valid_progress_bar = tqdm(total=int(self.datasets['valid'].num_rows/self.batch_size))
        valid_progress_bar.set_description(f"validation")

        context = {"subset": "train"} if self.model.training else {"subset": "valid"}
        for batch in dataloader:
            batch = {k: v.to(Device.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            gold_labels = batch["labels"]
            predictions = torch.argmax(outputs.logits, dim=-1)

            for metric in self.metrics:
                metric(gold_labels, predictions)

            valid_progress_bar.update(1)

        metrics = {metric.name: metric.get_metric() for metric in self.metrics}
        self.tracker.track(metrics, context=context)

        valid_progress_bar.set_description(f"validation | {', '.join([f'{k}: {v}' for k, v in metrics.items()])}")
        valid_progress_bar.close()


    def description(self) -> str:
        return "Adding a classification head on top of a pre-trained model. Freezing the pre-trained network."
