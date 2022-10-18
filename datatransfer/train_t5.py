import json
import os
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .naive_t5 import T5FineTuner
from .prepare_dataset import IeDataset



# 首先定义随机数种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


# 接下来设置日志工具
logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))


# 模型相关的参数
args_dict = dict(
    output_dir="t5_weiboner", # path to save the checkpoints
    model_name_or_path='google/mt5-large',
    tokenizer_name_or_path='google/mt5-large',
    max_seq_length=512,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=8,
    eval_batch_size=8,
    num_train_epochs=5,
    gradient_accumulation_steps=16,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False, # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)
args = argparse.Namespace(**args_dict)
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    precision= 16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
)

# 定义模型与训练器，然后开始训练
model = T5FineTuner(args)
trainer = pl.Trainer(**train_params)
trainer.fit(model)

model.model.save_pretrained('mt5_large_weiboner')


# 评价
dataset = IeDataset(model.tokenizer, 'test')
loader = DataLoader(dataset, batch_size=32)
model.model.eval()
outputs, targets = [], []
for batch in tqdm(loader):
    outs = model.model.generate(input_ids=batch['source_ids'].cuda(),
                                attention_mask=batch['source_mask'].cuda(),
                                max_length=20)
    dec = [model.tokenizer.decode(ids) for ids in outs]
    target = [model.tokenizer.decode(ids) for ids in batch['target_ids']]

    outputs.extend(dec)
    targets.extend(target)
json.dump([outputs, targets], open('model_output.json', 'w', encoding='utf-8'))

print('over.')