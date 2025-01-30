from samay.model import LPTMModel
from samay.dataset import LPTMDataset

repo = "lptm"
config = {
    "context_len": 512,
    "horizon_len": 192,
    "backend": "gpu",
    "per_core_batch_size": 32,
    "domain": "electricity",
}

lptm = LPTMModel(config=config, repo=repo)

train_dataset = LPTMDataset(name="electricity", datetime_col='date', path='data/ETTh1.csv', 
                              mode='train', context_len=config["context_len"], horizon_len=128)
val_dataset = LPTMDataset(name="electricity", datetime_col='date', path='data/ETTh1.csv',
                                mode='test', context_len=config["context_len"], horizon_len=config["horizon_len"])

avg_loss, trues, preds, histories = lptm.evaluate(val_dataset)