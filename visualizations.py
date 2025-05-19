import matplotlib.pyplot as plt
import pandas as pd
import os

data = {
    "Benchmark/Models": ["ETT1", "ETT2", "Flu-US", "PEM-Bays", "NY-Bike Demand", "NY-Taxi Demand", "Nasdaq", "M4"],
    "TimesFM": [0.58, 0.49, 1.32, 3.7, 2.8, 12.19, 0.22, 1.07],
    "Chronos": [0.59, 0.52, 1.21, 3.7, 3.1, 12.82, 0.27, 1.04],
    "MOIRAI": [0.62, 0.55, 1.31, 3.9, 3.5, 13.71, 0.24, 1.21],
    "Lag-LLAMA": [0.64, 0.57, 1.46, 3.9, 2.9, 13.43, 0.28, 1.33],
    "LPTM": [0.49, 0.46, 0.79, 2.5, 2.4, 11.84, 0.17, 0.94]
}
df = pd.DataFrame(data)
output_folder = "benchmark_visualizations"
os.makedirs(output_folder, exist_ok=True)

plt.style.use('ggplot')
plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.dpi': 100
})

def save_individual_benchmark_graphs():
    for benchmark in df["Benchmark/Models"]:
        subset = df[df["Benchmark/Models"] == benchmark].drop("Benchmark/Models", axis=1)
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(subset.columns, subset.values[0], edgecolor='black', linewidth=0.8)

        for bar, label in zip(bars, subset.columns):
            if label == "LPTM":
                bar.set_facecolor('#2ca02c')

        ax.set_xlabel("Models")
        ax.set_ylabel("MAE")
        ax.set_title(f"{benchmark} Performance", fontweight='bold')
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(rotation=30)
        fig.tight_layout()
        save_path = os.path.join(output_folder, f"{benchmark}_performance.png")
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved: {save_path}")

def save_combined_benchmark_graph():
    fig, ax = plt.subplots(figsize=(12, 8))
    df_plot = df.set_index("Benchmark/Models")
    df_plot.plot(kind="bar", ax=ax, edgecolor='black', linewidth=0.8)

    for container, label in zip(ax.containers, df_plot.columns):
        for bar in container:
            if label == "LPTM":
                bar.set_hatch("////")

    ax.set_xlabel("Benchmark Datasets")
    ax.set_ylabel("MAE")
    ax.set_title("Model Performance Across Benchmarks", fontweight='bold')
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(rotation=30)
    ax.legend(title="Models", loc='upper left')
    fig.tight_layout()
    save_path = os.path.join(output_folder, "All_Benchmarks_Comparison.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

save_individual_benchmark_graphs()
save_combined_benchmark_graph()
