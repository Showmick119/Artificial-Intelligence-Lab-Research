# Artificial Intelligence Lab — Research Contributions

This repository hosts all benchmarking, evaluation, and fine-tuning experiments on large pre-trained time-series models (LPTMs) under Dr. Prakash’s supervision.

## Benchmark Datasets
I evaluated Foundational Time-Series Models on eight standard datasets:
- **ETT1**  
- **ETT2**  
- **Flu-US**  
- **PEM-Bays**  
- **NY-Bike Demand**  
- **NY-Taxi Demand**  
- **Nasdaq**  
- **M4**  

## Model Suite & Comparisons
I compared and fine-tuned the following Foundational Time-Series Models:
1. [LPTM](https://arxiv.org/abs/2311.11413)  
2. [MOMENT](https://arxiv.org/abs/2402.03885)  
3. [TimesFM](https://arxiv.org/html/2310.10688v2)  
4. [Chronos](https://arxiv.org/abs/2403.07815)  
5. [MOIRAI](https://arxiv.org/abs/2402.02592)  
6. [TinytTimeMixers](https://arxiv.org/abs/2401.03955)  

_All experiments included systematic fine-tuning on each model to assess adaptability across datasets._

## Backend API
In `backend/`, you’ll find a RESTful Flask API deployed on a private NVIDIA DGX server using reverse-proxy and continuous testing with Postman & Ngrok. It supports:
- Model loading & versioning  
- Dataset uploads  
- On-the-fly fine-tuning  
- Inference endpoints  

## Data & Visualizations
- **Data**: Raw benchmark datasets live in `data/`.  
- **Visualizations**: All performance charts and plots are saved under `benchmark_visualizations/`.  

## Experiments
Detailed notebooks and logs of each experimental run are in `experiments/`. These include:
- Training/fine-tuning configurations  
- Metrics tracking (MAE, RMSE, etc.)  
- Ablation studies and hyperparameter sweeps  
