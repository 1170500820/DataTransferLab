import sys
sys.path.append('..')

from tqdm import tqdm
import json

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from datatransfer.naive_t5_2 import T5FineTuner
from datatransfer.prepare_dataset import DuIE_Dataset


conf = dict(
    dev_path='../data/prompted/duie_dev.jsonl',
    checkpoint_path='',

)

def predict(config):
    model = T5FineTuner.load_from_checkpoint(config['checkpoint_path'])
    dataset = DuIE_Dataset(model.tokenizer, data_type='dev')
    loader = DataLoader(dataset, batch_size=4)
    model.model.eval()
    outputs, targets = [], []
    print(f'the model device: {str(model.model.device)}')
    model.model.to('cuda')
    for batch in tqdm(loader):
        outs = model.model.generate(
            input_ids=batch['source_ids'].cuda(),
            attention_mask=batch['source_mask'].cuda()
        )
        dec = [model.tokenizer.decode(ids) for ids in outs]
        target = [model.tokenizer.decode(ids) for ids in batch['target_ids']]
        outputs.extend(dec)
        targets.extend(target)
    json.dump([outputs, targets], open('model_output.json', 'w', encoding='utf-8'), ensure_ascii=False)


if __name__ == '__main__':
    predict(conf)