import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Store the performance data in a pandas DataFrame
data = {
    "Model": ["LR", "DT", "KNN", "RF", "XGB", "MLP-LSTM"],
    "Validation_MSE": [339.42, 318.65, 296.02, 141.39, 158.79, 267.16],
    "Validation_MAE": [14.07, 10.39, 12.34, 8.01, 9.06, 12.24],
    "Validation_RMSE": [18.42, 17.85, 17.21, 11.89, 12.60, 16.35],
    "Validation_MAPE": [17.81, 12.38, 14.83, 9.57, 10.95, 15.51],
    "Validation_R2": [0.86, 0.86, 0.87, 0.94, 0.93, 0.88],
    "Test_MSE": [352.39, 309.34, 297.28, 143.16, 160.57, 246.25],
    "Test_MAE": [14.48, 10.49, 12.37, 8.14, 9.16, 11.48],
    "Test_RMSE": [18.77, 17.59, 17.24, 11.97, 12.67, 15.69],
    "Test_MAPE": [18.38, 12.35, 14.89, 9.76, 11.12, 13.98],
    "Test_R2": [0.85, 0.87, 0.87, 0.94, 0.93, 0.90],
}

df = pd.DataFrame(data)

# 2. List of metrics to plot
metrics = ["MSE", "MAE", "RMSE", "MAPE", "R2"]
models = df["Model"]

# 3. Loop through each metric to create a separate plot
for metric in metrics:
    val_scores = df[f"Validation_{metric}"]
    test_scores = df[f"Test_{metric}"]

    x = np.arange(len(models))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, val_scores, width, label='Validation', color='skyblue')
    rects2 = ax.bar(x + width/2, test_scores, width, label='Test', color='sandybrown')

    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Scores')
    ax.set_title(f'Model Performance Comparison: {metric}')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()

    # Save the figure to a file
    plt.savefig(f"{metric}_performance_chart.png", dpi=300)
    print(f"Generated chart: {metric}_performance_chart.png")
    plt.show()

# You can use plt.show() to display the plots if running in an interactive environment
# plt.show()