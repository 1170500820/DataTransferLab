"""
naive_t5.py的代码不适应新的版本
本代码为基于最新的pytorch_lightning对naive_t5.py的重写
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datatransfer.prepare_dataset import get_dataset
import pytorch_lightning as pl


from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)


class T5FineTuner(pl.LightningModule):
    def __init__(self, params=None):
        super(T5FineTuner, self).__init__()
        if params is not None:
            self.hparams.update(params)

        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.model_name)

    def forward(
            self,
            input_ids, attention_mask=None,
            decoder_input_ids=None, decoder_attention_mask=None,
            lm_labels=None
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch['target_ids']
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch['source_ids'],
            attention_mask=batch['source_mask'],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('loss', float(loss))
        return loss

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        #return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
#        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

#    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
#        optimizer.step()
#        optimizer.zero_grad()
#        self.lr_scheduler.step()

    def optimizer_step(self,
            epoch=None, batch_idx=None, optimizer=None, optimizer_idx=None,
            optimizer_closure=None, on_tpu=None, using_native_amp=None, using_lbfgs=None
            ):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()


    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, data_type='train')
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True,
                                num_workers=0)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpus)))
                // self.hparams.accumulate_grad_batches
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, data_type="dev")
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=0)

