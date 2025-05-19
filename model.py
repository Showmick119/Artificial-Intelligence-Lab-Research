import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from einops import rearrange
from jaxtyping import Float
from sklearn.metrics import mean_squared_error
from torchvision import transforms

from samay.dataset import MoiraiDataset

# from chronos import ChronosPipeline
from samay.models.chronosforecasting.chronos.chronos import ChronosPipeline
from samay.moirai_utils import convert_module_kwargs, filter_dict
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai.finetune import MoiraiFinetune
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from uni2ts.module.norm import RMSNorm

# from gluonts.model.forecast import Forecast, QuantileForecast, SampleForecast
from uni2ts.module.position import (
    BinaryAttentionBias,
    LearnedEmbedding,
    LearnedProjection,
)

# For moirai finetuning
from uni2ts.module.ts_embed import MultiInSizeLinear, MultiOutSizeLinear

from .metric import *
from .models.chronosforecasting.chronos.chronos import ChronosConfig, ChronosPipeline
from .models.chronosforecasting.chronos.chronos_bolt import (
    ChronosBoltConfig,
    ChronosBoltPipeline,
)
from .models.lptm.model.backbone import LPTMPipeline
from .models.moment.momentfm.models.moment import MOMENTPipeline
from .models.moment.momentfm.utils.masking import Masking
from .models.timesfm import timesfm as tfm
from .models.timesfm.timesfm import pytorch_patched_decoder as ppd
from .models.TinyTimeMixer.models.tinytimemixer.modeling_tinytimemixer import (
    TinyTimeMixerForPrediction,
)

from .models.Time_MoE.time_moe.models.modeling_time_moe import TimeMoeForPrediction
from .models.Time_MoE.time_moe.models.configuration_time_moe import TimeMoeConfig

from .utils import get_least_used_gpu, visualize


class Basemodel:
    def __init__(self, config=None, repo=None):
        """
        Args:
            config: dict, model configuration
            repo: str, Huggingface model repository id
        """
        self.config = config
        least_used_gpu = get_least_used_gpu()
        if least_used_gpu >= 0:
            self.device = torch.device(f"cuda:{least_used_gpu}")
        else:
            self.device = torch.device("cpu")

    def finetune(self, dataset, **kwargs):
        raise NotImplementedError

    def forecast(self, input, **kwargs):
        raise NotImplementedError

    def evaluate(self, dateset, **kwargs):
        pass

    def save(self, path):
        pass


class TimesfmModel(Basemodel):
    def __init__(self, config=None, repo=None, ckpt=None, **kwargs):
        """
        Args:
            config: dict, model configuration
            repo: str, Huggingface model repository id
        """
        super().__init__(config=config, repo=repo)
        hparams = tfm.TimesFmHparams(**self.config)
        if repo:
            try:
                ckpt = tfm.TimesFmCheckpoint(huggingface_repo_id=repo)
            except:
                ckpt = None
                raise ValueError(f"Repository {repo} not found")

        self.model = tfm.TimesFm(hparams=hparams, checkpoint=ckpt)

    def finetune(self, dataset, freeze_transformer=True, **kwargs):
        """
        Args:
            dataset: dataset for finetuning, call get_data_loader() to get the dataloader
            freeze_transformer: bool, whether to freeze the transformer layers
        Returns:
            FinetuneModel: ppd.PatchedDecoderFinetuneModel, finetuned model
        """
        lr = 1e-4 if "lr" not in kwargs else kwargs["lr"]
        epoch = 5 if "epoch" not in kwargs else kwargs["epoch"]

        core_layer_tpl = self.model._model
        # Todo: whether add freq
        FinetunedModel = ppd.PatchedDecoderFinetuneModel(core_layer_tpl=core_layer_tpl)
        if freeze_transformer:
            for param in FinetunedModel.core_layer.stacked_transformer.parameters():
                param.requires_grad = False
        FinetunedModel.to(self.device)
        FinetunedModel.train()
        dataloader = dataset.get_data_loader()
        optimizer = torch.optim.Adam(FinetunedModel.parameters(), lr=lr)

        avg_loss = 0
        for epoch in range(epoch):
            for i, (inputs) in enumerate(dataloader):
                inputs = dataset.preprocess(inputs)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                optimizer.zero_grad()
                outputs = FinetunedModel.compute_predictions(
                    inputs, train_horizon_len=self.config["horizon_len"]
                )  # b, n, seq_len, 1+quantiles
                loss = FinetunedModel.compute_loss(outputs, inputs)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
            avg_loss /= len(dataloader)
            print(f"Epoch {epoch}, Loss: {avg_loss}")

        self.model._model = FinetunedModel.core_layer
        return self.model

    def forecast(self, input, **kwargs):
        """
        Args:
            input: torch.Tensor, input data
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - the mean forecast of size (# inputs, # forecast horizon),
                - the full forecast (mean + quantiles) of size
                (# inputs,  # forecast horizon, 1 + # quantiles).
        """
        return self.model.forecast(input)

    def plot(self, dataset, **kwargs):
        """
        Plot the forecast results.
        Args:
            dataset: dataset for plotting, call get_data_loader() to get the dataloader
        """
        dataloader = dataset.get_data_loader()
        trues, preds, histories, losses = [], [], [], []
        with torch.no_grad():
            for i, (inputs) in enumerate(dataloader):
                inputs = dataset.preprocess(inputs)
                input_ts = inputs["input_ts"]
                input_ts = np.squeeze(input_ts, axis=0)
                actual_ts = inputs["actual_ts"].detach().cpu().numpy()
                actual_ts = np.squeeze(actual_ts, axis=0)

                output, _ = self.model.forecast(input_ts)
                output = output[:, 0 : actual_ts.shape[1]]

                loss = np.mean((output - actual_ts) ** 2)
                losses.append(loss.item())
                trues.append(actual_ts)
                preds.append(output)
                histories.append(input_ts)

        losses = np.array(losses)
        average_loss = np.average(losses)
        trues = np.concatenate(trues, axis=0).reshape(
            -1, dataset.num_ts, trues[-1].shape[-1]
        )
        preds = np.concatenate(preds, axis=0).reshape(
            -1, dataset.num_ts, preds[-1].shape[-1]
        )
        histories = np.concatenate(histories, axis=0).reshape(
            -1, dataset.num_ts, histories[-1].shape[-1]
        )

        visualize(
            task_name="forecasting",
            trues=trues,
            preds=preds,
            history=histories,
            **kwargs,
        )

        # return average_loss, trues, preds, histories

    def evaluate(self, dataset, **kwargs):
        """
        Evaluate the model.
        Args:
            dataset: dataset for evaluation, call get_data_loader() to get the dataloader
        Returns:
            Dict[str, float]: evaluation metrics, including mse, mae, mase, mape, rmse, nrmse, smape, msis, nd, mwsq, crps
        """
        dataloader = dataset.get_data_loader()
        trues, preds, histories, quantiles, losses = [], [], [], [], []

        with torch.no_grad():
            for i, (inputs) in enumerate(dataloader):
                inputs = dataset.preprocess(inputs)
                input_ts = inputs["input_ts"]
                input_ts = np.squeeze(input_ts, axis=0)
                actual_ts = inputs["actual_ts"].detach().cpu().numpy()
                actual_ts = np.squeeze(actual_ts, axis=0)

                output, quantile_output = self.model.forecast(input_ts)
                output = output[:, 0 : actual_ts.shape[1]]
                quantile_output = quantile_output[:, 0 : actual_ts.shape[1]]

                loss = np.mean((output - actual_ts) ** 2)
                losses.append(loss.item())
                trues.append(actual_ts)
                preds.append(output)
                histories.append(input_ts)
                quantiles.append(quantile_output)

        losses = np.array(losses)
        average_loss = np.average(losses)
        trues = np.concatenate(trues, axis=0).reshape(
            -1, dataset.num_ts, trues[-1].shape[-1]
        )
        preds = np.concatenate(preds, axis=0).reshape(
            -1, dataset.num_ts, preds[-1].shape[-1]
        )
        histories = np.concatenate(histories, axis=0).reshape(
            -1, dataset.num_ts, histories[-1].shape[-1]
        )
        quantiles = np.concatenate(quantiles, axis=0).reshape(
            quantiles[-1].shape[-1], -1, dataset.num_ts, quantiles[-1].shape[-2]
        )

        mse = MSE(trues, preds)
        mae = MAE(trues, preds)
        mase = MASE(trues, preds)
        mape = MAPE(trues, preds)
        rmse = RMSE(trues, preds)
        nrmse = NRMSE(trues, preds)
        smape = SMAPE(trues, preds)
        msis = MSIS(trues, preds)
        nd = ND(trues, preds)
        mwsq = MWSQ(trues, preds, quantiles)
        crps = CRPS(trues, preds, quantiles)

        return {
            "mse": mse,
            "mae": mae,
            "mase": mase,
            "mape": mape,
            "rmse": rmse,
            "nrmse": nrmse,
            "smape": smape,
            "msis": msis,
            "nd": nd,
            "mwsq": mwsq,
            "crps": crps,
        }


class ChronosModel(Basemodel):
    def __init__(self, config=None, repo=None):
        """
        Args:
            config: dict, model configuration
            repo: str, Huggingface model repository id
        """
        super().__init__(config=config, repo=repo)
        if repo:
            print("Loading Chronos model from Huggingface repository")
            try:
                self.pipeline = ChronosPipeline.from_pretrained(
                    repo, device_map=self.device
                )
            except:
                raise ValueError(f"Repository {repo} not found")
        else:
            print("Initializing a new Chronos model without pre-trained weights")
            self.pipeline = ChronosPipeline(config=ChronosConfig(**config))

    def finetune(self, dataset, **kwargs):
        """
        Args:
            dataset: dataset for finetuning, call get_data_loader() to get the dataloader
        """
        # Todo: finetune model
        finetune_model = self.pipeline.model.model
        dataloader = dataset.get_data_loader()
        finetune_model.to(self.device)
        finetune_model.train()
        optimizer = torch.optim.AdamW(finetune_model.parameters(), lr=1e-4)

        avg_loss = 0

        for epoch in range(5):
            for i, data in enumerate(dataloader):
                input_ids = data["input_ids"].to(self.device)
                ids_shape = input_ids.shape
                input_ids = input_ids.reshape(ids_shape[0] * ids_shape[1], ids_shape[2])
                attention_mask = data["attention_mask"].to(self.device)
                mask_shape = attention_mask.shape
                attention_mask = attention_mask.reshape(
                    mask_shape[0] * mask_shape[1], mask_shape[2]
                )
                labels = data["labels"].to(self.device)
                label_shape = labels.shape
                labels = labels.reshape(label_shape[0] * label_shape[1], label_shape[2])
                optimizer.zero_grad()
                output = finetune_model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = output.loss
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
            avg_loss /= len(dataloader)
            print(f"Epoch {epoch}, Loss: {avg_loss}")

        finetune_model.eval()

    def plot(self, dataset, horizon_len, quantile_levels, **kwargs):
        """
        Plot the forecast results.
        Args:
            dataset: dataset for plotting, call get_data_loader() to get the dataloader
            horizon_len: int, forecast horizon length
            quantile_levels: list, list of quantile levels
        """
        # Todo: forecast
        dataloader = dataset.get_data_loader()
        trues, preds, histories = [], [], []
        for i, data in enumerate(dataloader):
            input_seq = data["input_seq"]
            forecast_seq = data["forecast_seq"]
            shape = input_seq.shape
            input_seq = input_seq.reshape(shape[0] * shape[1], shape[2])
            input_seq = torch.tensor(input_seq)
            quantiles, mean = self.pipeline.predict_quantiles(
                context=input_seq,
                prediction_length=horizon_len,
                quantile_levels=quantile_levels,
            )
            trues.append(forecast_seq.detach().cpu().numpy())
            mean = mean.reshape(
                forecast_seq.shape[0], forecast_seq.shape[1], forecast_seq.shape[2]
            )
            preds.append(mean.detach().cpu().numpy())
            input_seq = input_seq.reshape(shape[0], shape[1], shape[2])
            histories.append(input_seq.detach().cpu().numpy())

        trues = np.concatenate(trues, axis=0)
        preds = np.concatenate(preds, axis=0)
        histories = np.concatenate(histories, axis=0)

        visualize(
            task_name="forecasting",
            trues=trues,
            preds=preds,
            history=histories,
            **kwargs,
        )

    def evaluate(self, dataset, horizon_len, quantile_levels, **kwargs):
        """
        Evaluate the model.
        Args:
            dataset: dataset for evaluation, call get_data_loader() to get the dataloader
            horizon_len: int, forecast horizon length
            quantile_levels: list, list of quantile levels
        Returns:
            Dict[str, float]: evaluation metrics, including mse, mae, mase, mape, rmse, nrmse, smape, msis, nd, mwsq, crps
        """
        dataloader = dataset.get_data_loader()
        trues, preds, histories, quantile_forecasts = [], [], [], []
        for i, data in enumerate(dataloader):
            input_seq = data["input_seq"]
            forecast_seq = data["forecast_seq"]
            shape = input_seq.shape
            input_seq = input_seq.reshape(shape[0] * shape[1], shape[2])
            input_seq = torch.tensor(input_seq)
            quantiles, mean = self.pipeline.predict_quantiles(
                context=input_seq,
                prediction_length=horizon_len,
                quantile_levels=quantile_levels,
                limit_prediction_length=False,
            )
            trues.append(forecast_seq.detach().cpu().numpy())
            mean = mean.reshape(
                forecast_seq.shape[0], forecast_seq.shape[1], forecast_seq.shape[2]
            )
            preds.append(mean.detach().cpu().numpy())
            quantiles = quantiles.reshape(
                quantiles.shape[-1],
                forecast_seq.shape[0],
                forecast_seq.shape[1],
                forecast_seq.shape[2],
            )
            quantile_forecasts.append(quantiles.detach().cpu().numpy())
            input_seq = input_seq.reshape(shape[0], shape[1], shape[2])
            histories.append(input_seq.detach().cpu().numpy())

        trues = np.concatenate(trues, axis=0)
        preds = np.concatenate(preds, axis=0)
        histories = np.concatenate(histories, axis=0)
        quantile_forecasts = np.concatenate(quantile_forecasts, axis=1)

        mse = MSE(trues, preds)
        mae = MAE(trues, preds)
        mase = MASE(trues, preds)
        mape = MAPE(trues, preds)
        rmse = RMSE(trues, preds)
        nrmse = NRMSE(trues, preds)
        smape = SMAPE(trues, preds)
        msis = MSIS(trues, preds)
        nd = ND(trues, preds)
        mwsq = MWSQ(trues, preds, quantile_forecasts)
        crps = CRPS(trues, preds, quantile_forecasts)

        return {
            "mse": mse,
            "mae": mae,
            "mase": mase,
            "mape": mape,
            "rmse": rmse,
            "nrmse": nrmse,
            "smape": smape,
            "msis": msis,
            "nd": nd,
            "mwsq": mwsq,
            "crps": crps,
        }


class ChronosBoltModel(Basemodel):
    def __init__(self, config=None, repo=None):
        """
        Args:
            config: dict, model configuration
            repo: str, Huggingface model repository id
        """
        super().__init__(config=config, repo=repo)
        if repo:
            print("Loading Chronos model from Huggingface repository")
            try:
                self.pipeline = ChronosBoltPipeline.from_pretrained(
                    repo, device_map=self.device
                )
            except:
                raise ValueError(f"Repository {repo} not found")
        else:
            print("Initializing a new Chronos model without pre-trained weights")
            self.pipeline = ChronosBoltPipeline(config=ChronosBoltConfig(**config))

    def finetune(self, dataset, **kwargs):
        """
        Args:
            dataset: dataset for finetuning, call get_data_loader() to get the dataloader
        """
        # Todo: finetune model
        finetune_model = self.pipeline.model
        dataloader = dataset.get_data_loader()
        finetune_model.to(self.device)
        finetune_model.train()
        optimizer = torch.optim.AdamW(finetune_model.parameters(), lr=1e-4)

        avg_loss = 0

        for epoch in range(10):
            for i, data in enumerate(dataloader):
                context, forecast = data
                context = context.to(self.device)
                forecast = forecast.to(self.device)
                c_shape = context.shape
                context = context.reshape(c_shape[0] * c_shape[1], c_shape[2])
                f_shape = forecast.shape
                forecast = forecast.reshape(f_shape[0] * f_shape[1], f_shape[2])
                optimizer.zero_grad()
                output = finetune_model(context=context, target=forecast)
                loss = output.loss
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
            avg_loss /= len(dataloader)
            print(f"Epoch {epoch}, Loss: {avg_loss}")

        finetune_model.eval()

    def plot(self, dataset, horizon_len, quantile_levels, **kwargs):
        """
        Plot the forecast results.
        Args:
            dataset: dataset for plotting, call get_data_loader() to get the dataloader
            horizon_len: int, forecast horizon length
            quantile_levels: list, list of quantile levels
        """
        dataloader = dataset.get_data_loader()
        trues, preds, histories = [], [], []
        for i, data in enumerate(dataloader):
            context, forecast_seq = data
            c_shape = context.shape
            context = context.reshape(c_shape[0] * c_shape[1], c_shape[2])
            context = torch.tensor(context)
            quantiles, mean = self.pipeline.predict_quantiles(
                context=context,
                prediction_length=horizon_len,
                quantile_levels=quantile_levels,
            )
            trues.append(forecast_seq.detach().cpu().numpy())
            mean = mean.reshape(
                forecast_seq.shape[0], forecast_seq.shape[1], forecast_seq.shape[2]
            )
            preds.append(mean.detach().cpu().numpy())
            input_seq = context.reshape(c_shape[0], c_shape[1], c_shape[2])
            histories.append(input_seq.detach().cpu().numpy())

        trues = np.concatenate(trues, axis=0)
        preds = np.concatenate(preds, axis=0)
        histories = np.concatenate(histories, axis=0)

        visualize(
            task_name="forecasting",
            trues=trues,
            preds=preds,
            history=histories,
            **kwargs,
        )

    def evaluate(self, dataset, horizon_len, quantile_levels, **kwargs):
        """
        Evaluate the model.
        Args:
            dataset: dataset for evaluation, call get_data_loader() to get the dataloader
            horizon_len: int, forecast horizon length
            quantile_levels: list, list of quantile levels
        Returns:
            Dict[str, float]: evaluation metrics, including mse, mae, mase, mape, rmse, nrmse, smape, msis, nd, mwsq, crps
        """
        dataloader = dataset.get_data_loader()
        trues, preds, histories, quantile_forecasts = [], [], [], []
        for i, data in enumerate(dataloader):
            context, forecast_seq = data
            c_shape = context.shape
            context = context.reshape(c_shape[0] * c_shape[1], c_shape[2])
            context = torch.tensor(context)
            quantiles, mean = self.pipeline.predict_quantiles(
                context=context,
                prediction_length=horizon_len,
                quantile_levels=quantile_levels,
            )
            trues.append(forecast_seq.detach().cpu().numpy())
            mean = mean.reshape(
                forecast_seq.shape[0], forecast_seq.shape[1], forecast_seq.shape[2]
            )
            preds.append(mean.detach().cpu().numpy())
            quantiles = quantiles.reshape(
                quantiles.shape[-1],
                forecast_seq.shape[0],
                forecast_seq.shape[1],
                forecast_seq.shape[2],
            )
            quantile_forecasts.append(quantiles.detach().cpu().numpy())
            input_seq = context.reshape(c_shape[0], c_shape[1], c_shape[2])
            histories.append(input_seq.detach().cpu().numpy())

        trues = np.concatenate(trues, axis=0)
        preds = np.concatenate(preds, axis=0)
        histories = np.concatenate(histories, axis=0)
        quantile_forecasts = np.concatenate(quantile_forecasts, axis=1)

        mse = MSE(trues, preds)
        mae = MAE(trues, preds)
        mase = MASE(trues, preds)
        mape = MAPE(trues, preds)
        rmse = RMSE(trues, preds)
        nrmse = NRMSE(trues, preds)
        smape = SMAPE(trues, preds)
        msis = MSIS(trues, preds)
        nd = ND(trues, preds)
        mwsq = MWSQ(trues, preds, quantile_forecasts)
        crps = CRPS(trues, preds, quantile_forecasts)

        return {
            "mse": mse,
            "mae": mae,
            "mase": mase,
            "mape": mape,
            "rmse": rmse,
            "nrmse": nrmse,
            "smape": smape,
            "msis": msis,
            "nd": nd,
            "mwsq": mwsq,
            "crps": crps,
        }


class LPTMModel(Basemodel):
    def __init__(self, config=None):
        super().__init__(config=config, repo=None)
        # config["patch_len"] = config["max_patch"]
        self.model = LPTMPipeline.from_pretrained(
            "kage08/lptm-large2", model_kwargs=self.config
        )

        self.model.init()

    def finetune(self, dataset, task_name="forecasting", **kwargs):
        # arguments
        max_lr = 1e-4 if "lr" not in kwargs else kwargs["lr"]
        max_epoch = 5 if "epoch" not in kwargs else kwargs["epoch"]
        max_norm = 5.0 if "norm" not in kwargs else kwargs["norm"]
        mask_ratio = 0.25 if "mask_ratio" not in kwargs else kwargs["mask_ratio"]

        if task_name == "imputation" or task_name == "detection":
            mask_generator = Masking(mask_ratio=mask_ratio)

        dataloader = dataset.get_data_loader()
        criterion = torch.nn.MSELoss()
        if task_name == "classification":
            criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=max_lr)
        criterion.to(self.device)
        scaler = torch.amp.GradScaler()

        total_steps = len(dataloader) * max_epoch
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3
        )
        self.model.to(self.device)
        self.model.train()

        for epoch in range(max_epoch):
            losses = []
            for i, data in enumerate(dataloader):
                # unpack the data
                if task_name == "forecasting":
                    timeseries, input_mask, forecast = data
                    # Move the data to the GPU
                    timeseries = timeseries.float().to(self.device)
                    input_mask = input_mask.to(self.device)
                    forecast = forecast.float().to(self.device)
                    with torch.amp.autocast(device_type="cuda"):
                        output = self.model(x_enc=timeseries, input_mask=input_mask)
                    loss = criterion(output.forecast, forecast)
                elif task_name == "forecasting2":
                    timeseries, input_mask, forecast = data
                    # Move the data to the GPU
                    timeseries = timeseries.float().to(self.device)
                    input_mask = input_mask.to(self.device)
                    forecast = forecast.float().to(self.device)
                    with torch.amp.autocast(device_type="cuda"):
                        output = self.model(x_enc=timeseries, input_mask=input_mask)
                    loss = criterion(
                        output.forecast[:, :, -forecast.shape[-1] :], forecast
                    )

                elif task_name == "imputation":
                    timeseries, input_mask = data
                    n_channels = timeseries.shape[1]
                    # Move the data to the GPU
                    timeseries = timeseries.float().to(self.device)
                    timeseries = timeseries.reshape(-1, 1, timeseries.shape[-1])
                    input_mask = input_mask.to(self.device).long()
                    input_mask = input_mask.repeat_interleave(n_channels, axis=0)
                    mask = (
                        mask_generator.generate_mask(
                            x=timeseries, input_mask=input_mask
                        )
                        .to(self.device)
                        .long()
                    )
                    output = self.model(
                        x_enc=timeseries, input_mask=input_mask, mask=mask
                    )
                    with torch.amp.autocast(device_type="cuda"):
                        recon_loss = criterion(output.reconstruction, timeseries)
                    observed_mask = input_mask * (1 - mask)
                    masked_loss = observed_mask * recon_loss
                    loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)

                elif task_name == "detection":
                    timeseries, input_mask, label = data
                    n_channels = timeseries.shape[1]
                    seq_len = timeseries.shape[-1]
                    timeseries = (
                        timeseries.reshape(-1, 1, seq_len).float().to(self.device)
                    )
                    input_mask = input_mask.to(self.device).long()
                    input_mask = input_mask.repeat_interleave(n_channels, axis=0)
                    mask = (
                        mask_generator.generate_mask(
                            x=timeseries, input_mask=input_mask
                        )
                        .to(self.device)
                        .long()
                    )
                    output = self.model(
                        x_enc=timeseries, input_mask=input_mask, mask=mask
                    )
                    with torch.amp.autocast(device_type="cuda"):
                        loss = criterion(output.reconstruction, timeseries)

                elif task_name == "classification":
                    timeseries, input_mask, label = data
                    timeseries = timeseries.to(self.device).float()
                    label = label.to(self.device).long()
                    output = self.model(x_enc=timeseries)
                    with torch.amp.autocast(device_type="cuda"):
                        loss = criterion(output.logits, label)

                optimizer.zero_grad(set_to_none=True)
                # Scales the loss for mixed precision training
                scaler.scale(loss).backward()

                # Clip gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()

                losses.append(loss.item())

            losses = np.array(losses)
            average_loss = np.average(losses)
            print(f"Epoch {epoch}: Train loss: {average_loss:.3f}")

            scheduler.step()

        return self.model

    def evaluate(self, dataset, task_name="forecasting"):
        dataloader = dataset.get_data_loader()
        criterion = torch.nn.MSELoss()
        self.model.to(self.device)
        self.model.eval()
        if task_name == "forecasting":
            trues, preds, histories, losses = [], [], [], []
            with torch.no_grad():
                for i, data in enumerate(dataloader):
                    # unpack the data
                    timeseries, input_mask, forecast = data
                    # Move the data to the GPU
                    timeseries = timeseries.float().to(self.device)
                    input_mask = input_mask.to(self.device)
                    forecast = forecast.float().to(self.device)

                    output = self.model(x_enc=timeseries, input_mask=input_mask)
                    loss = criterion(output.forecast, forecast)
                    losses.append(loss.item())
                    trues.append(forecast.detach().cpu().numpy())
                    preds.append(output.forecast.detach().cpu().numpy())
                    histories.append(timeseries.detach().cpu().numpy())

            losses = np.array(losses)
            average_loss = np.average(losses)
            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)
            histories = np.concatenate(histories, axis=0)

            return average_loss, trues, preds, histories

        elif task_name == "forecasting2":
            trues, preds, histories, losses = [], [], [], []
            with torch.no_grad():
                for i, data in enumerate(dataloader):
                    # unpack the data
                    timeseries, input_mask, forecast = data
                    # Move the data to the GPU
                    timeseries = timeseries.float().to(self.device)
                    input_mask = input_mask.to(self.device)
                    forecast = forecast.float().to(self.device)

                    output = self.model(x_enc=timeseries, input_mask=input_mask)
                    loss = criterion(
                        output.forecast[:, :, -forecast.shape[-1] :], forecast
                    )
                    losses.append(loss.item())
                    trues.append(forecast.detach().cpu().numpy())
                    preds.append(output.forecast.detach().cpu().numpy())
                    histories.append(timeseries.detach().cpu().numpy())

            losses = np.array(losses)
            average_loss = np.average(losses)
            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)
            histories = np.concatenate(histories, axis=0)

            mse = MSE(trues, preds)
            mae = MAE(trues, preds)
            mase = MASE(trues, preds)
            mape = MAPE(trues, preds)
            rmse = RMSE(trues, preds)
            nrmse = NRMSE(trues, preds)
            smape = SMAPE(trues, preds)
            msis = MSIS(trues, preds)
            nd = ND(trues, preds)

            return {
                "mse": mse,
                "mae": mae,
                "mase": mase,
                "mape": mape,
                "rmse": rmse,
                "nrmse": nrmse,
                "smape": smape,
                "msis": msis,
                "nd": nd,
            }

            # return average_loss, trues, preds, histories

        elif task_name == "imputation":
            trues, preds, masks = [], [], []
            mask_generator = Masking(mask_ratio=0.25)
            with torch.no_grad():
                for i, data in enumerate(dataloader):
                    # unpack the data
                    timeseries, input_mask = data
                    trues.append(timeseries.numpy())
                    n_channels = timeseries.shape[1]
                    # Move the data to the GPU
                    timeseries = timeseries.float().to(self.device)
                    timeseries = timeseries.reshape(-1, 1, timeseries.shape[-1])
                    # print(input_mask.shape)
                    input_mask = input_mask.to(self.device).long()
                    input_mask = input_mask.repeat_interleave(n_channels, axis=0)
                    # print(timeseries.shape, input_mask.shape)
                    mask = (
                        mask_generator.generate_mask(
                            x=timeseries, input_mask=input_mask
                        )
                        .to(self.device)
                        .long()
                    )
                    output = self.model(
                        x_enc=timeseries, input_mask=input_mask, mask=mask
                    )
                    reconstruction = output.reconstruction.reshape(
                        -1, n_channels, timeseries.shape[-1]
                    )
                    mask = mask.reshape(-1, n_channels, timeseries.shape[-1])
                    preds.append(reconstruction.detach().cpu().numpy())
                    masks.append(mask.detach().cpu().numpy())

            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)
            masks = np.concatenate(masks, axis=0)

            return trues, preds, masks

        elif task_name == "detection":
            trues, preds, labels = [], [], []
            with torch.no_grad():
                for i, data in enumerate(dataloader):
                    # unpack the data
                    timeseries, input_mask, label = data
                    timeseries = timeseries.to(self.device).float()
                    input_mask = input_mask.to(self.device).long()
                    label = label.to(self.device).long()
                    output = self.model(x_enc=timeseries, input_mask=input_mask)

                    trues.append(timeseries.detach().cpu().numpy())
                    preds.append(output.reconstruction.detach().cpu().numpy())
                    labels.append(label.detach().cpu().numpy())

            trues = np.concatenate(trues, axis=0).flatten()
            preds = np.concatenate(preds, axis=0).flatten()
            labels = np.concatenate(labels, axis=0).flatten()

            return trues, preds, labels

        elif task_name == "classification":
            accuracy = 0
            total = 0
            embeddings = []
            labels = []
            with torch.no_grad():
                for i, data in enumerate(dataloader):
                    # unpack the data
                    timeseries, input_mask, label = data
                    timeseries = timeseries.to(self.device).float()
                    label = label.to(self.device).long()
                    labels.append(label.detach().cpu().numpy())
                    input_mask = input_mask.to(self.device).long()
                    output = self.model(x_enc=timeseries, input_mask=input_mask)
                    embedding = output.embeddings.mean(dim=1)
                    embeddings.append(embedding.detach().cpu().numpy())
                    _, predicted = torch.max(output.logits, 1)
                    total += label.size(0)
                    accuracy += (predicted == label).sum().item()

            accuracy = accuracy / total
            embeddings = np.concatenate(embeddings)
            labels = np.concatenate(labels)
            return accuracy, embeddings, labels


class MomentModel(Basemodel):
    def __init__(self, config=None, repo=None):
        """
        Args:
            config: dict, model configuration
            repo: str, Huggingface model repository id
        """
        super().__init__(config=config, repo=repo)
        if not repo:
            # raise ValueError("Moment model requires a repository")
            print("Initializing a new MOMENT model without pre-trained weights")
            base_config = json.load(
                open("/nethome/sli999/TSFMProject/config/moment_base.json", "r")
            )
            self.model = MOMENTPipeline(config=base_config, model_kwargs=self.config)
        else:
            print(f"Loading MOMENT model from {repo}")
            self.model = MOMENTPipeline.from_pretrained(repo, model_kwargs=self.config)
        self.model.init()

    def finetune(self, dataset, task_name="forecasting", **kwargs):
        """
        Args:
            dataset: dataset for finetuning, call get_data_loader() to get the dataloader
            task_name: str, task name, forecasting, imputation, detection, classification
        Returns:
            MOMENT model
        """
        # arguments
        max_lr = 1e-4 if "lr" not in kwargs else kwargs["lr"]
        max_epoch = 5 if "epoch" not in kwargs else kwargs["epoch"]
        max_norm = 1.0 if "norm" not in kwargs else kwargs["norm"]
        mask_ratio = 0.25 if "mask_ratio" not in kwargs else kwargs["mask_ratio"]

        if task_name == "imputation" or task_name == "detection":
            mask_generator = Masking(mask_ratio=mask_ratio)

        dataloader = dataset.get_data_loader()
        criterion = torch.nn.MSELoss()
        if task_name == "classification":
            criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=max_lr)
        criterion.to(self.device)
        scaler = torch.amp.GradScaler()

        total_steps = len(dataloader) * max_epoch
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=0.3
        )
        self.model.to(self.device)
        self.model.train()

        for epoch in range(max_epoch):
            losses = []
            for i, data in enumerate(dataloader):
                # unpack the data
                if task_name == "forecasting":
                    timeseries, input_mask, forecast = data
                    # Move the data to the GPU
                    timeseries = timeseries.float().to(self.device)
                    input_mask = input_mask.to(self.device)
                    forecast = forecast.float().to(self.device)
                    # with torch.amp.autocast(device_type='cuda'):
                    output = self.model(x_enc=timeseries, input_mask=input_mask)
                    loss = criterion(output.forecast, forecast)

                elif task_name == "imputation":
                    timeseries, input_mask = data
                    n_channels = timeseries.shape[1]
                    # Move the data to the GPU
                    timeseries = timeseries.float().to(self.device)
                    timeseries = timeseries.reshape(-1, 1, timeseries.shape[-1])
                    input_mask = input_mask.to(self.device).long()
                    input_mask = input_mask.repeat_interleave(n_channels, axis=0)
                    mask = (
                        mask_generator.generate_mask(
                            x=timeseries, input_mask=input_mask
                        )
                        .to(self.device)
                        .long()
                    )
                    output = self.model(
                        x_enc=timeseries, input_mask=input_mask, mask=mask
                    )
                    # with torch.amp.autocast(device_type='cuda'):
                    recon_loss = criterion(output.reconstruction, timeseries)
                    observed_mask = input_mask * (1 - mask)
                    masked_loss = observed_mask * recon_loss
                    loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)

                elif task_name == "detection":
                    timeseries, input_mask, label = data
                    n_channels = timeseries.shape[1]
                    seq_len = timeseries.shape[-1]
                    timeseries = (
                        timeseries.reshape(-1, 1, seq_len).float().to(self.device)
                    )
                    input_mask = input_mask.to(self.device).long()
                    input_mask = input_mask.repeat_interleave(n_channels, axis=0)
                    mask = (
                        mask_generator.generate_mask(
                            x=timeseries, input_mask=input_mask
                        )
                        .to(self.device)
                        .long()
                    )
                    output = self.model(
                        x_enc=timeseries, input_mask=input_mask, mask=mask
                    )
                    # with torch.amp.autocast(device_type='cuda'):
                    loss = criterion(output.reconstruction, timeseries)

                elif task_name == "classification":
                    timeseries, input_mask, label = data
                    timeseries = timeseries.to(self.device).float()
                    label = label.to(self.device).long()
                    output = self.model(x_enc=timeseries)
                    # with torch.amp.autocast(device_type='cuda'):
                    loss = criterion(output.logits, label)

                optimizer.zero_grad(set_to_none=True)
                # Scales the loss for mixed precision training
                scaler.scale(loss).backward()

                # Clip gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()

                losses.append(loss.item())

            losses = np.array(losses)
            average_loss = np.average(losses)
            print(f"Epoch {epoch}: Train loss: {average_loss:.3f}")

            scheduler.step()

        return self.model

    def plot(self, dataset, task_name="forecasting"):
        """
        Plot the forecast results.
        Args:
            dataset: dataset for plotting, call get_data_loader() to get the dataloader
            task_name: str, task name, forecasting, imputation, detection, classification
        """
        dataloader = dataset.get_data_loader()
        criterion = torch.nn.MSELoss()
        self.model.to(self.device)
        self.model.eval()
        if task_name == "forecasting":
            trues, preds, histories, losses = [], [], [], []
            with torch.no_grad():
                for i, data in enumerate(dataloader):
                    # unpack the data
                    timeseries, input_mask, forecast = data
                    # Move the data to the GPU
                    timeseries = timeseries.float().to(self.device)
                    input_mask = input_mask.to(self.device)
                    forecast = forecast.float().to(self.device)

                    output = self.model(x_enc=timeseries, input_mask=input_mask)
                    loss = criterion(output.forecast, forecast)
                    losses.append(loss.item())
                    trues.append(forecast.detach().cpu().numpy())
                    preds.append(output.forecast.detach().cpu().numpy())
                    histories.append(timeseries.detach().cpu().numpy())

            losses = np.array(losses)
            average_loss = np.average(losses)
            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)
            histories = np.concatenate(histories, axis=0)

            visualize(
                task_name="forecasting", trues=trues, preds=preds, history=histories
            )

            # return average_loss, trues, preds, histories

        elif task_name == "imputation":
            trues, preds, masks = [], [], []
            mask_generator = Masking(mask_ratio=0.25)
            with torch.no_grad():
                for i, data in enumerate(dataloader):
                    # unpack the data
                    timeseries, input_mask = data
                    trues.append(timeseries.numpy())
                    n_channels = timeseries.shape[1]
                    # Move the data to the GPU
                    timeseries = timeseries.float().to(self.device)
                    timeseries = timeseries.reshape(-1, 1, timeseries.shape[-1])
                    # print(input_mask.shape)
                    input_mask = input_mask.to(self.device).long()
                    input_mask = input_mask.repeat_interleave(n_channels, axis=0)
                    # print(timeseries.shape, input_mask.shape)
                    mask = (
                        mask_generator.generate_mask(
                            x=timeseries, input_mask=input_mask
                        )
                        .to(self.device)
                        .long()
                    )
                    output = self.model(
                        x_enc=timeseries, input_mask=input_mask, mask=mask
                    )
                    reconstruction = output.reconstruction.reshape(
                        -1, n_channels, timeseries.shape[-1]
                    )
                    mask = mask.reshape(-1, n_channels, timeseries.shape[-1])
                    preds.append(reconstruction.detach().cpu().numpy())
                    masks.append(mask.detach().cpu().numpy())

            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)
            masks = np.concatenate(masks, axis=0)

            visualize(task_name="imputation", trues=trues, preds=preds, masks=masks)

            # return trues, preds, masks

        elif task_name == "detection":
            trues, preds, labels = [], [], []
            with torch.no_grad():
                for i, data in enumerate(dataloader):
                    # unpack the data
                    timeseries, input_mask, label = data
                    timeseries = timeseries.to(self.device).float()
                    input_mask = input_mask.to(self.device).long()
                    label = label.to(self.device).long()
                    output = self.model(x_enc=timeseries, input_mask=input_mask)

                    trues.append(timeseries.detach().cpu().numpy())
                    preds.append(output.reconstruction.detach().cpu().numpy())
                    labels.append(label.detach().cpu().numpy())

            trues = np.concatenate(trues, axis=0).flatten()
            preds = np.concatenate(preds, axis=0).flatten()
            labels = np.concatenate(labels, axis=0).flatten()

            visualize(task_name="detection", trues=trues, preds=preds, labels=labels)

            # return trues, preds, labels

        # elif task_name == "classification":
        #     accuracy = 0
        #     total = 0
        #     embeddings = []
        #     labels = []
        #     with torch.no_grad():
        #         for i, data in enumerate(dataloader):
        #             # unpack the data
        #             timeseries, input_mask, label = data
        #             timeseries = timeseries.to(self.device).float()
        #             label = label.to(self.device).long()
        #             labels.append(label.detach().cpu().numpy())
        #             input_mask = input_mask.to(self.device).long()
        #             output = self.model(x_enc=timeseries, input_mask=input_mask)
        #             embedding = output.embeddings.mean(dim=1)
        #             embeddings.append(embedding.detach().cpu().numpy())
        #             _, predicted = torch.max(output.logits, 1)
        #             total += label.size(0)
        #             accuracy += (predicted == label).sum().item()

        #     accuracy = accuracy / total
        #     embeddings = np.concatenate(embeddings)
        #     labels = np.concatenate(labels)
        #     return accuracy, embeddings, labels

    def evaluate(self, dataset, task_name="forecasting"):
        """
        Evaluate the model.
        Args:
            dataset: dataset for evaluation, call get_data_loader() to get the dataloader
            task_name: str, task name, forecasting, imputation, detection, classification
        Returns:
            Dict[str, float]: evaluation metrics, including mse, mae, mase, mape, rmse, nrmse, smape, msis, nd, mwsq, crps
        """
        dataloader = dataset.get_data_loader()
        self.model.to(self.device)
        self.model.eval()
        if task_name == "forecasting":
            criterion = torch.nn.MSELoss()
            trues, preds, histories, losses = [], [], [], []
            with torch.no_grad():
                for i, data in enumerate(dataloader):
                    # unpack the data
                    timeseries, input_mask, forecast = data
                    # Move the data to the GPU
                    timeseries = timeseries.float().to(self.device)
                    input_mask = input_mask.to(self.device)
                    forecast = forecast.float().to(self.device)

                    output = self.model(x_enc=timeseries, input_mask=input_mask)
                    loss = criterion(output.forecast, forecast)
                    losses.append(loss.item())
                    trues.append(forecast.detach().cpu().numpy())
                    preds.append(output.forecast.detach().cpu().numpy())
                    histories.append(timeseries.detach().cpu().numpy())

            losses = np.array(losses)
            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)
            histories = np.concatenate(histories, axis=0)

            mse = MSE(trues, preds)
            mae = MAE(trues, preds)
            mase = MASE(trues, preds)
            mape = MAPE(trues, preds)
            rmse = RMSE(trues, preds)
            nrmse = NRMSE(trues, preds)
            smape = SMAPE(trues, preds)
            msis = MSIS(trues, preds)
            nd = ND(trues, preds)

            return {
                "mse": mse,
                "mae": mae,
                "mase": mase,
                "mape": mape,
                "rmse": rmse,
                "nrmse": nrmse,
                "smape": smape,
                "msis": msis,
                "nd": nd,
            }

        elif task_name == "classification":
            accuracy = 0
            total = 0
            embeddings = []
            labels = []
            with torch.no_grad():
                for i, data in enumerate(dataloader):
                    # unpack the data
                    timeseries, input_mask, label = data
                    timeseries = timeseries.to(self.device).float()
                    label = label.to(self.device).long()
                    labels.append(label.detach().cpu().numpy())
                    input_mask = input_mask.to(self.device).long()
                    output = self.model(x_enc=timeseries, input_mask=input_mask)
                    embedding = output.embeddings.mean(dim=1)
                    embeddings.append(embedding.detach().cpu().numpy())
                    _, predicted = torch.max(output.logits, 1)
                    total += label.size(0)
                    accuracy += (predicted == label).sum().item()

            accuracy = accuracy / total
            embeddings = np.concatenate(embeddings)
            labels = np.concatenate(labels)
            return accuracy, embeddings, labels


class TinyTimeMixerModel(Basemodel):
    def __init__(self, config=None, repo=None):
        """
        Args:
            config: dict, model configuration
            repo: str, Huggingface model repository id
        """
        super().__init__(config=config, repo=repo)
        if repo:
            context_len = config["context_len"]
            horizon_len = config["horizon_len"]
            horizon_list = [96, 192, 336, 720]
            closest_larger_horizon = min([x for x in horizon_list if x >= horizon_len])
            if context_len == 512 and closest_larger_horizon == 96:
                revision = "main"
            else:
                revision = f"{context_len}-{closest_larger_horizon}-r2"
            self.model = TinyTimeMixerForPrediction.from_pretrained(
                repo, revision=revision, prediction_filter_length=horizon_len
            )
            # self.model = self.model.to(self.device)
        else:
            raise ValueError("TinyTimeMixer model requires a repository")

    def finetune(self, dataset, **kwargs):
        """
        Args:
            dataset: dataset for finetuning, call get_data_loader() to get the dataloader
        """
        dataloader = dataset.get_data_loader()
        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        for epoch in range(5):
            total_loss = 0
            for i, data in enumerate(dataloader):
                context, forecast_seq = data
                context = context.float().permute(0, 2, 1).to(self.device)
                forecast_seq = forecast_seq.float().permute(0, 2, 1).to(self.device)
                optimizer.zero_grad()
                output = self.model(past_values=context, future_values=forecast_seq)
                loss = output.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch}, Loss: {avg_loss}")
        self.model.eval()

    def plot(self, dataset, **kwargs):
        """
        Plot the forecast results.
        Args:
            dataset: dataset for plotting, call get_data_loader() to get the dataloader
        """
        dataloader = dataset.get_data_loader()
        trues, preds, histories = [], [], []
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                context, forecast_seq = data
                context = context.float().permute(0, 2, 1).to(self.device)
                forecast_seq = forecast_seq.float().permute(0, 2, 1).to(self.device)
                output = self.model(past_values=context, future_values=forecast_seq)
                pred = output.prediction_outputs
                trues.append(forecast_seq.permute(0, 2, 1).detach().cpu().numpy())
                preds.append(pred.permute(0, 2, 1).detach().cpu().numpy())
                histories.append(context.permute(0, 2, 1).detach().cpu().numpy())

            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)
            histories = np.concatenate(histories, axis=0)

        visualize(
            task_name="forecasting",
            trues=trues,
            preds=preds,
            history=histories,
            **kwargs,
        )

    def evaluate(self, dataset, **kwargs):
        """
        Evaluate the model.
        Args:
            dataset: dataset for evaluation, call get_data_loader() to get the dataloader
        Returns:
            Dict[str, float]: evaluation metrics, including mse, mae, mase, mape, rmse, nrmse, smape, msis, nd
        """
        dataloader = dataset.get_data_loader()
        trues, preds, histories = [], [], []
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                context, forecast_seq = data
                context = context.float().permute(0, 2, 1).to(self.device)
                forecast_seq = forecast_seq.float().permute(0, 2, 1).to(self.device)
                output = self.model(past_values=context, future_values=forecast_seq)
                pred = output.prediction_outputs
                trues.append(forecast_seq.permute(0, 2, 1).detach().cpu().numpy())
                preds.append(pred.permute(0, 2, 1).detach().cpu().numpy())
                histories.append(context.permute(0, 2, 1).detach().cpu().numpy())

            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)
            histories = np.concatenate(histories, axis=0)

        print(trues.shape, preds.shape, histories.shape)
        mse = MSE(trues, preds)
        mae = MAE(trues, preds)
        mase = MASE(trues, preds)
        mape = MAPE(trues, preds)
        rmse = RMSE(trues, preds)
        nrmse = NRMSE(trues, preds)
        smape = SMAPE(trues, preds)
        msis = MSIS(trues, preds)
        nd = ND(trues, preds)

        return {
            "mse": mse,
            "mae": mae,
            "mase": mase,
            "mape": mape,
            "rmse": rmse,
            "nrmse": nrmse,
            "smape": smape,
            "msis": msis,
            "nd": nd,
        }


class MoiraiTSModel(Basemodel):
    def __init__(
        self,
        config=None,
        repo=None,
        model_type="moirai-moe",
        model_size="small",
        **kwargs,
    ):
        super().__init__(config=config, repo=repo)
        # config.get(<key>, <default_value> if key not found)
        self.horizon_len = config.get("horizon_len", 32)
        self.context_len = config.get("context_len", 128)
        self.patch_size = config.get("patch_size", 16)
        self.batch_size = config.get("batch_size", 16)
        self.num_samples = config.get("num_samples", 100)
        self.target_dim = config.get("target_dim", 1)
        self.feat_dynamic_real_dim = config.get("feat_dynamic_real_dim", 0)
        self.past_feat_dynamic_real_dim = config.get("past_feat_dynamic_real_dim", 0)
        self.model_type = model_type
        self.finetuned_model = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type == "moirai":  # standard moirai
            if repo is None:
                repo = f"Salesforce/moirai-1.1-R-{model_size}"
            self.repo = repo

            self.model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(self.repo),
                prediction_length=self.horizon_len,
                context_length=self.context_len,
                patch_size=self.patch_size,
                num_samples=self.num_samples,
                target_dim=self.target_dim,
                feat_dynamic_real_dim=self.feat_dynamic_real_dim,
                past_feat_dynamic_real_dim=self.past_feat_dynamic_real_dim,
            )

        elif model_type == "moirai-moe":  # moirai with Mixture of Experts
            if repo is None:
                repo = f"Salesforce/moirai-moe-1.0-R-{model_size}"
            self.repo = repo

            self.model = MoiraiMoEForecast(
                module=MoiraiMoEModule.from_pretrained(self.repo),
                prediction_length=self.horizon_len,
                context_length=self.context_len,
                patch_size=self.patch_size,
                num_samples=self.num_samples,
                target_dim=self.target_dim,
                feat_dynamic_real_dim=self.feat_dynamic_real_dim,
                past_feat_dynamic_real_dim=self.past_feat_dynamic_real_dim,
            )
        self.model.to(self.device)

    def preprocess_inputs(self, inputs: dict):
        """Preprocess the inputs to the model - specifically adds the following fields:
        +--------------------+--------------------------------------+-----------------------+----------------------------------+
        | FIELD              | DESCRIPTION                          | TYPE                  | SHAPE                            |
        +--------------------+--------------------------------------+-----------------------+----------------------------------+
        | target             | Batched time series data             | torch.tensor[float]   | (batch_size, seq_len, max_patch) |
        | observed_mask      | Binary mask for the context part     | torch.tensor[bool]    | (batch_size, seq_len, max_patch) |
        | prediction_mask    | Binary mask for the prediction part  | torch.tensor[bool]    | (batch_size, seq_len)            |
        | time_id            | Time index                           | torch.tensor[int]     | (batch_size, seq_len)            |
        | sample_id          | Time index                           | torch.tensor[int]     | (batch_size, seq_len)            |
        | variate_id         | Index indicating the variate         | torch.tensor[int]     | (batch_size, seq_len)            |
        | patch_size         | Patch size the model should use      | torch.tensor[int]     | (batch_size, seq_len)            |
        +--------------------+--------------------------------------+-----------------------+----------------------------------+

        Args:
            inputs (dict): Dictionary containing the input data.

        Returns:
            dict: Preprocessed input data.
        """
        (target, observed_mask, sample_id, time_id, variate_id, prediction_mask) = (
            self.model._convert(
                patch_size=self.patch_size,
                past_target=inputs["past_target"],
                past_observed_target=inputs["past_observed_target"],
                past_is_pad=inputs["past_is_pad"],
            )
        )
        inputs["target"] = target
        inputs["observed_mask"] = observed_mask
        inputs["sample_id"] = sample_id
        inputs["time_id"] = time_id
        inputs["variate_id"] = variate_id
        inputs["prediction_mask"] = prediction_mask
        inputs["patch_size"] = torch.tensor(
            np.full(shape=sample_id.shape, fill_value=self.patch_size, dtype=np.int64),
            dtype=torch.int64,
        )

        return inputs

    def _format_preds(
        self,
        preds: Float[torch.Tensor, "sample batch combine_seq patch"],
        context_token_len: int,
        pred_token_len: int,
    ) -> Float[torch.Tensor, "batch sample future_time *tgt"]:
        start = self.target_dim * context_token_len
        end = start + self.target_dim * pred_token_len
        preds = preds[..., start:end, : self.patch_size]
        preds = rearrange(
            preds,
            "sample ... (dim seq) patch -> ... sample (seq patch) dim",
            dim=self.target_dim,
        )[..., : self.horizon_len, :]
        return preds.squeeze(-1)

    def evaluate(
        self,
        dataset: MoiraiDataset,
        metrics: list[str] = ["MSE"],
        output_transforms: transforms.Compose = None,
        num_sample_flag: bool = False,
        zero_shot: bool = True,
        leaderboard: bool = False,
        **kwargs,
    ):
        """For a given test dataset, we evaluate the model using the given metrics.

        Args:
            dataset (MoiraiDataset): Dataset to evaluate the model on.
            metrics (list, optional): Metrics you want to evaluate the model on. Defaults to ["MSE"].
            output_transforms (transforms.Compose, optional): A set of transforms to be applied on the model output. Defaults to None.
            num_sample_flage (bool, optional): If True, the model will use number of samples to sample from the distribution for forecasting. Defaults to False.
            zero_shot (bool, optional): If True, the standard model will be used, else the finetuned model will be used. Defaults to True.
            leaderboard (bool, optional): If True, only the metrics will be returned. Defaults to False.

        Raises:
            ValueError: Any metric other than "MSE" or "MASE is not supported.

        Returns:
            dict: Evaluation results for each column (variate).
            dict: True values for each column (variate).
            dict: Predictions for each column (variate).
            dict: Histories for each column (variate).
        """
        # required fields for the forecast
        inp_names = [
            "past_target",
            "past_observed_target",
            "past_is_pad",
        ]
        if self.feat_dynamic_real_dim > 0:
            inp_names.extend(["feat_dynamic_real", "observed_feat_dynamic_real"])
        if self.past_feat_dynamic_real_dim > 0:
            inp_names.extend(
                ["past_feat_dynamic_real", "past_observed_feat_dynamic_real"]
            )

        # get the batched data
        inference_loader = dataset.get_dataloader()

        # set model in eval mode
        self.model.eval()

        # predict
        forecast = []
        with torch.no_grad():
            for batch in inference_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                inputs = filter_dict(batch, inp_names)
                if (
                    zero_shot
                ):  # Finetune forward asks for keys like target, observed_mask, etc.
                    outputs = (
                        self.model.forward(**inputs).detach().cpu().numpy()
                    )  # convert the tensor output to numpy array
                elif not zero_shot and self.finetuned_model is not None:
                    # get the context and prediction token lengths
                    context_token_len = math.ceil(self.context_len / self.patch_size)
                    pred_token_len = math.ceil(self.horizon_len / self.patch_size)
                    num_context_tokens = context_token_len * self.target_dim
                    num_pred_tokens = pred_token_len * self.target_dim

                    # prepare the inputs
                    pred_index = torch.arange(
                        start=context_token_len - 1,
                        end=num_context_tokens,
                        step=context_token_len,
                    )
                    assign_index = torch.arange(
                        start=num_context_tokens,
                        end=num_context_tokens + num_pred_tokens,
                        step=pred_token_len,
                    )

                    old_keys = list(inputs.keys())
                    inputs = self.preprocess_inputs(inputs)
                    new_keys = list(inputs.keys())
                    inputs = filter_dict(inputs, list(set(new_keys) - set(old_keys)))
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            inputs[k] = v.to(self.device)

                    # get the forecast
                    if pred_token_len == 1:
                        distr = self.finetuned_model.forward(**inputs)
                        preds = distr.sample(torch.Size((self.num_samples,)))
                        preds[..., assign_index, :] = preds[..., pred_index, :]
                        outputs = self._format_preds(
                            preds=preds,
                            context_token_len=context_token_len,
                            pred_token_len=pred_token_len,
                        )
                    else:
                        distr = self.finetuned_model.forward(**inputs)
                        preds = distr.sample(torch.Size((self.num_samples,)))

                        expand_target = (
                            inputs["target"]
                            .unsqueeze(0)
                            .repeat(self.num_samples, 1, 1, 1)
                        )
                        expand_prediction_mask = (
                            inputs["prediction_mask"]
                            .unsqueeze(0)
                            .repeat(self.num_samples, 1, 1)
                        )
                        expand_observed_mask = (
                            inputs["observed_mask"]
                            .unsqueeze(0)
                            .expand(self.num_samples, -1, -1, -1)
                        )
                        expand_sample_id = (
                            inputs["sample_id"]
                            .unsqueeze(0)
                            .expand(self.num_samples, -1, -1)
                        )
                        expand_time_id = (
                            inputs["time_id"]
                            .unsqueeze(0)
                            .expand(self.num_samples, -1, -1)
                        )
                        expand_variate_id = (
                            inputs["variate_id"]
                            .unsqueeze(0)
                            .expand(self.num_samples, -1, -1)
                        )
                        expand_patch_size = (
                            inputs["patch_size"]
                            .unsqueeze(0)
                            .expand(self.num_samples, -1, -1)
                        )

                        expand_target[..., assign_index, :] = preds[..., pred_index, :]
                        expand_prediction_mask[..., assign_index] = False

                        remain_step = pred_token_len - 1
                        while remain_step > 0:
                            distr = self.finetuned_model.forward(
                                expand_target,
                                expand_observed_mask,
                                expand_sample_id,
                                expand_time_id,
                                expand_variate_id,
                                expand_prediction_mask,
                                expand_patch_size,
                            )
                            preds = distr.sample(torch.Size((1,)))
                            _, _, bs, token, ps = preds.shape
                            preds = preds.view(-1, bs, token, ps)

                            pred_index = assign_index
                            assign_index = assign_index + 1
                            expand_target[..., assign_index, :] = preds[
                                ..., pred_index, :
                            ]
                            expand_prediction_mask[..., assign_index] = False

                            remain_step -= 1

                        outputs = self._format_preds(
                            preds=expand_target,
                            context_token_len=context_token_len,
                            pred_token_len=pred_token_len,
                        )

                # Apply output transforms
                if output_transforms is not None:
                    outputs = output_transforms(outputs)

                # sample if needed
                if num_sample_flag:
                    num_collected_samples = outputs[0].shape[0]
                    collected_samples = [outputs]
                    # do so until we have enough samples
                    while num_collected_samples < self.num_samples:
                        outputs = self.model.forward(**inputs).detach().cpu().numpy()
                        # Apply output transforms
                        if output_transforms is not None:
                            outputs = output_transforms(outputs)
                        collected_samples.append(outputs)
                        num_collected_samples += outputs[0].shape[0]
                    # stack the collected samples
                    outputs = np.stack(
                        [
                            np.concatenate(s)[: self.num_samples]
                            for s in zip(*collected_samples)
                        ]
                    )
                    # assert that we have the right number of samples
                    assert len(outputs[0]) == self.num_samples, (
                        "We do not have enough samples"
                    )

                forecast.extend([np.array(x) for x in outputs.tolist()])

        # Iterators for input, label and forecast
        print("Forecasting done....now testing")
        input_it = iter(dataset.dataset.input)
        label_it = iter(dataset.dataset.label)
        forecast_it = iter(forecast)
        # Quantile levels obtained from cli/conf/eval/default.yaml of MOIRAI repository
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        trues = {}
        preds = {}
        histories = {}
        quantile_preds = {}
        eval_windows = []

        with torch.no_grad():  # No need to compute gradients
            # Iterate over each window
            for input, label, forecast in zip(input_it, label_it, forecast_it):
                true_values = (
                    label["target"].squeeze()
                    if isinstance(label["target"], np.ndarray)
                    else np.array(label["target"])
                )
                past_values = (
                    input["target"].squeeze()
                    if isinstance(input["target"], np.ndarray)
                    else np.array(input["target"])
                )
                quantiles = np.percentile(
                    forecast[:, : min(self.horizon_len, true_values.shape[0])],
                    [q * 100 for q in quantile_levels],
                    axis=0,
                )
                pred_values = np.median(forecast, axis=0)[
                    : min(self.horizon_len, true_values.shape[0])
                ]  # Median of the forecasted values

                length = len(past_values)

                eval = []
                for metric in metrics:
                    if metric == "MSE":
                        eval.append(mean_squared_error(true_values, pred_values))

                    # MASE = current model's MAE / naive model's MAE
                    elif metric == "MASE":
                        forecast_error = np.mean(np.abs(true_values - pred_values))
                        naive_error = np.mean(
                            np.abs(true_values[1:] - true_values[:-1])
                        )
                        if naive_error == 0:  # Avoid division by zero
                            eval.append(np.inf)
                        else:
                            eval.append(forecast_error / naive_error)
                    else:
                        raise ValueError(f"Unsupported metric: {metric}")
                eval_windows.append(eval)

                # Update history, true values and predictions
                if length not in histories.keys():
                    histories[length] = []
                    trues[length] = []
                    preds[length] = []
                    quantile_preds[length] = []
                histories[length].append(past_values)
                trues[length].append(true_values)
                preds[length].append(pred_values)
                quantile_preds[length].append(quantiles)

        eval_windows = np.mean(np.array(eval_windows), axis=0)
        eval_results = {}
        for i in range(len(metrics)):
            eval_results[metrics[i]] = eval_windows[i]

        # Convert to numpy arrays
        histories = [np.array(histories[key]) for key in histories.keys()]
        trues = [np.array(trues[key]) for key in trues.keys()]
        preds = [np.array(preds[key]) for key in preds.keys()]
        quantile_preds = [
            np.array(quantile_preds[key]) for key in quantile_preds.keys()
        ]
        quantile_preds = [np.transpose(q, (1, 0, 2)) for q in quantile_preds]

        mse = np.mean(np.array([MSE(t, p) for t, p in zip(trues, preds)]), axis=0)
        mae = np.mean(np.array([MAE(t, p) for t, p in zip(trues, preds)]), axis=0)
        mase = np.mean(np.array([MASE(t, p) for t, p in zip(trues, preds)]), axis=0)
        mape = np.mean(np.array([MAPE(t, p) for t, p in zip(trues, preds)]), axis=0)
        rmse = np.mean(np.array([RMSE(t, p) for t, p in zip(trues, preds)]), axis=0)
        nrmse = np.mean(np.array([NRMSE(t, p) for t, p in zip(trues, preds)]), axis=0)
        smape = np.mean(np.array([SMAPE(t, p) for t, p in zip(trues, preds)]), axis=0)
        msis = np.mean(np.array([MSIS(t, p) for t, p in zip(trues, preds)]), axis=0)
        nd = np.mean(np.array([ND(t, p) for t, p in zip(trues, preds)]), axis=0)

        mwsq = np.mean(
            np.array([MWSQ(t, p, q) for t, p, q in zip(trues, preds, quantile_preds)]),
            axis=0,
        )
        crps = np.mean(
            np.array([CRPS(t, p, q) for t, p, q in zip(trues, preds, quantile_preds)]),
            axis=0,
        )

        leaderboard_metrics = {
            "mse": mse,
            "mae": mae,
            "mase": mase,
            "mape": mape,
            "rmse": rmse,
            "nrmse": nrmse,
            "smape": smape,
            "msis": msis,
            "nd": nd,
            "mwsq": mwsq,
            "crps": crps,
        }

        if leaderboard:
            return leaderboard_metrics
        else:
            return leaderboard_metrics, trues, preds, histories

    def finetune(self, dataset, **kwargs):
        """Finetune the model on the given dataset.

        Args:
            dataset (MoiraiDataset): Dataset containing the input data and relevant functions like dataloaders etc.

        Returns:
            _type_: _description_
        """
        model_size = self.repo.split("-")[-1]
        if self.model_type == "moirai":
            model_config = (
                f"../src/uni2ts/cli/conf/finetune/model/moirai_1.1_R_{model_size}.yaml"
            )
        elif self.model_type == "moirai-moe":
            model_config = f"../src/uni2ts/cli/conf/finetune/model/moirai_moe_1.0_R_{model_size}.yaml"

        with open(model_config, "r") as file:
            fin_model_config = yaml.safe_load(file)

        # lr = 1e-4 if "lr" not in fin_model_config else float(fin_model_config["lr"])
        lr = 5e-6
        weight_decay = (
            1e-1
            if "weight_decay" not in fin_model_config
            else float(fin_model_config["weight_decay"])
        )
        self.batch_size = (
            kwargs["batch_size"] if "batch_size" in kwargs else self.batch_size
        )
        epochs = 5
        assert epochs <= kwargs["max_epochs"], (
            "epochs should be less than or equal to max_epochs"
        )

        # Number of batches per epoch required for calculating the number of training steps
        num_batches = len(dataset.dataset) // self.batch_size
        if (
            "num_batches_per_epoch" in kwargs.keys()
        ):  # If num_batches_per_epoch is provided
            num_batches_per_epoch = kwargs["num_batches_per_epoch"]
            epochs = min(epochs, num_batches // num_batches_per_epoch)
        else:
            num_batches_per_epoch = num_batches // epochs

        training_steps = num_batches_per_epoch * kwargs["max_epochs"]
        module_args = convert_module_kwargs(
            fin_model_config["module_kwargs"]
        )  # remove _target_ fields
        self.patch_size = self.model.module.in_proj.in_features_ls[
            0
        ]  # update patch_size

        # Trainer configuration (from uni2ts/cli/train.py)
        # mod_torch is the trainer configuration without _target_ fields or any key
        # whose value is neither a list or dictionary
        if kwargs["tf32"]:
            assert kwargs["mod_torch"]["precision"] == 32, (
                "Precision should be 32 for tf32"
            )
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # For now, self.model.module.patch_sizes i just [16] from the config file
        # But in finetune, we are using patch_sizes as [8,16,32,64,128]
        # So, we need to update the patch_sizes in the model
        # self.model.module.patch_sizes = list(module_args["patch_sizes"])

        # Load the model
        FinetunedModel = MoiraiFinetune(
            min_patches=fin_model_config["min_patches"],
            min_mask_ratio=fin_model_config["min_mask_ratio"],
            max_mask_ratio=fin_model_config["max_mask_ratio"],
            max_dim=fin_model_config["max_dim"],
            num_training_steps=training_steps,
            num_warmup_steps=fin_model_config["num_warmup_steps"],
            module_kwargs=module_args,
            beta1=fin_model_config["beta1"],
            beta2=fin_model_config["beta2"],
            val_metric=fin_model_config["val_metric"],
            weight_decay=fin_model_config["weight_decay"],
            model_type=self.model_type,
            model_size=model_size,
        )

        # Pytorch version
        FinetunedModel.to(self.device)
        FinetunedModel.train()  # Set model to training mode

        # Freeze the transformer layers
        # First we finetune the whole model - Not good
        # Freeze the last two encoder layers and param_proj
        for mn, m in FinetunedModel.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue

                fpn = f"{mn}.{pn}" if mn else pn
                # print(f"Checking fpn before freezing: {fpn}")

                # Freeze everything except the last 2 encoder layers and param_proj
                if fpn.split(".")[1] in [
                    "in_proj",
                    "res_proj",
                    "feat_proj",
                ]:  # Freeze all initial layers
                    p.requires_grad = False
                elif fpn.split(".")[1] == "encoder":
                    if (
                        len(fpn.split(".")) > 3
                        and fpn.split(".")[2] == "layers"
                        and int(fpn.split(".")[3]) < 5
                    ):  # Freeze all but last two encoder layers
                        p.requires_grad = False

        # Load the dataset
        dataloader = (
            dataset.get_dataloader()
        )  # look at if mode=="train" case for more info

        decay = set()
        no_decay = set()

        whitelist_params = (
            LearnedProjection,
            MultiInSizeLinear,
            MultiOutSizeLinear,
            nn.Linear,
        )
        blacklist_params = (
            BinaryAttentionBias,
            LearnedEmbedding,
            RMSNorm,
            nn.Embedding,
            nn.LayerNorm,
        )

        for mn, m in FinetunedModel.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue

                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_params):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_params):
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {
            pn: p for pn, p in FinetunedModel.named_parameters() if p.requires_grad
        }

        optim_groups = [
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [v for k, v in param_dict.items() if k in (list(no_decay))],
                ),
                "weight_decay": weight_decay,
            },
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [v for k, v in param_dict.items() if k not in (list(no_decay))],
                ),
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=lr,
            betas=(FinetunedModel.hparams.beta1, FinetunedModel.hparams.beta2),
            eps=1e-6,
        )

        # scheduler = get_scheduler(
        #     SchedulerType.COSINE_WITH_RESTARTS,
        #     optimizer,
        #     num_warmup_steps=FinetunedModel.hparams.num_warmup_steps,
        #     num_training_steps=FinetunedModel.hparams.num_training_steps,
        # )

        loss_vals = []
        for epoch in range(epochs):
            avg_loss = 0
            for i, (inputs) in enumerate(dataloader):  # each batch is processed
                inputs = self.preprocess_inputs(
                    inputs
                )  # patchify and other fields added
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                        if v.dtype == torch.float32 or v.dtype == torch.float64:
                            inputs[k] = inputs[k].requires_grad_()
                optimizer.zero_grad()  # reset gradients
                # distribution of predictions
                torch.autograd.set_detect_anomaly(True)
                outputs = FinetunedModel.forward(
                    target=inputs["target"],
                    observed_mask=inputs["observed_mask"],
                    sample_id=inputs["sample_id"],
                    time_id=inputs["time_id"],
                    variate_id=inputs["variate_id"],
                    prediction_mask=inputs["prediction_mask"],
                    patch_size=inputs["patch_size"],
                )
                loss = FinetunedModel.hparams.loss_func(
                    pred=outputs,
                    **{
                        field: inputs[field]
                        for field in [
                            "target",
                            "prediction_mask",
                            "observed_mask",
                            "sample_id",
                            "variate_id",
                        ]
                    },
                )

                loss.backward()
                optimizer.step()
                # scheduler.step()
                avg_loss += loss.item()
            avg_loss /= len(dataloader)
            print(f"Epoch {epoch}: Loss: {avg_loss:.3f}")
            loss_vals.append(avg_loss)
        print("Finetuning done")

        # Plot the loss values
        plt.grid(True)
        plt.plot(loss_vals)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.savefig(f"training_loss_{epochs}_epochs_wo_scheduler.png")

        self.finetuned_model = FinetunedModel
        print("Fineuned model updated")


class TimeMoEModel(Basemodel):
    def __init__(self, config=None, repo=None, **kwargs):
        super().__init__(config=config, repo=repo)
        if repo:
            self.model = TimeMoeForPrediction.from_pretrained(repo)
        else:
            t_config = TimeMoeConfig(**self.config)
            self.model = TimeMoeForPrediction(t_config)

    def finetune(self, dataset, **kwargs):
        """
        Finetune the model on the given dataset.
        Args:
            dataset: dataset for finetuning
        """
        # Implement finetuning logic here
        dataloader = dataset.get_data_loader()
        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        for epoch in range(5):
            total_loss = 0
            for i, data in enumerate(dataloader):
                context, forecast_seq, loss_mask = data
                context = context.float().to(self.device)
                forecast_seq = forecast_seq.float().to(self.device)
                loss_mask = loss_mask.float().to(self.device)
                optimizer.zero_grad()
                output = self.model(input_ids=context, labels=forecast_seq, loss_masks=loss_mask)
                loss = output.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        self.model.eval()


    def plot(self, dataset, **kwargs):
        """
        Plot the results of the model on the given dataset.
        Args:
            dataset: dataset for plotting
        """
        # Implement plotting logic here
        dataloader = dataset.get_data_loader()
        self.model.to(self.device)
        self.model.eval()
        trues, preds, histories = [], [], []
        with torch.no_grad():
            for data in dataloader:
                context, forecast_seq = data
                context = context.float().to(self.device)
                forecast_seq = forecast_seq.float().to(self.device)
                output = self.model.generate(inputs=context, max_new_tokens=forecast_seq.shape[1])
                pred = output[:, -forecast_seq.shape[1]:]
                pred = pred.cpu().numpy()
                true = forecast_seq.cpu().numpy()
                history = context.cpu().numpy()
                trues.append(true)
                preds.append(pred)
                histories.append(history)
        trues = np.concatenate(trues, axis=0).reshape(-1, dataset.n_channels, dataset.horizon_len)
        preds = np.concatenate(preds, axis=0).reshape(-1, dataset.n_channels, dataset.horizon_len)
        histories = np.concatenate(histories, axis=0).reshape(-1, dataset.n_channels, dataset.context_len)
        
        visualize(
            task_name="forecasting",
            trues=trues,
            preds=preds,
            history=histories,
        )
        
    def evaluate(self, dataset, **kwargs):
        """
        Evaluate the model on the given dataset.
        Args:
            dataset: dataset for evaluation
        """
        # Implement evaluation logic here
        dataloader = dataset.get_data_loader()
        self.model.to(self.device)
        self.model.eval()
        trues, preds, histories = [], [], []

        with torch.no_grad():
            for data in dataloader:
                context, forecast_seq = data
                context = context.float().to(self.device)
                forecast_seq = forecast_seq.float().to(self.device)
                output = self.model.generate(inputs=context, max_new_tokens=forecast_seq.shape[1])
                pred = output[:, -forecast_seq.shape[1]:]
                pred = pred.cpu().numpy()
                true = forecast_seq.cpu().numpy()
                history = context.cpu().numpy()
                trues.append(true)
                preds.append(pred)
                histories.append(history)
        trues = np.concatenate(trues, axis=0).reshape(-1, dataset.n_channels, dataset.horizon_len)
        preds = np.concatenate(preds, axis=0).reshape(-1, dataset.n_channels, dataset.horizon_len)
        histories = np.concatenate(histories, axis=0).reshape(-1, dataset.n_channels, dataset.context_len)

        # Calculate metrics
        mse = MSE(trues, preds)
        mae = MAE(trues, preds)
        mase = MASE(trues, preds)
        mape = MAPE(trues, preds)
        rmse = RMSE(trues, preds)
        nrmse = NRMSE(trues, preds)
        smape = SMAPE(trues, preds)
        msis = MSIS(trues, preds)
        nd = ND(trues, preds)

        return {
            "mse": mse,
            "mae": mae,
            "mase": mase,
            "mape": mape,
            "rmse": rmse,
            "nrmse": nrmse,
            "smape": smape,
            "msis": msis,
            "nd": nd,
        }
        


if __name__ == "__main__":
    name = "timesfm"
    repo = "google/timesfm-1.0-200m-pytorch"
    config = {
        "context_len": 128,
        "horizon_len": 32,
        "backend": "gpu",
        "per_core_batch_size": 32,
        "input_patch_len": 32,
        "output_patch_len": 128,
        "num_layers": 20,
        "model_dims": 1280,
        "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    }

    tfm_model = TimesfmModel(config=config, repo=repo)
    model = tfm_model.model
    # print(tfm.model)
    df = pd.read_csv("/nethome/sli999/data/Tycho/dengue_laos.csv")
    df = df[df["SourceName"] == "Laos Dengue Surveillance System"]
    df = df[["Admin1ISO", "PeriodStartDate", "CountValue"]]
    df.columns = ["unique_id", "ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(by=["unique_id", "ds"])
    forecast_df = model.forecast_on_df(
        inputs=df,
        freq="D",  # daily frequency
        value_name="y",
        num_jobs=1,
    )
    forecast_df = forecast_df[["ds", "unique_id", "timesfm"]]
    forecast_df.columns = ["ds", "unique_id", "y"]

    print(forecast_df.head())
