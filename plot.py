"""Plot the results"""

import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric_over_step(df, metric, plot_all: bool = False):
    legal_values = ["val_pplxs", "val_accs", "val_losses", "train_accs", "train_losses"]
    assert metric in legal_values, f"The metric must be one of {legal_values}"

    df_lin = df[df["linear_value"]]
    df_nonlin = df[~df["linear_value"]]

    lin_vals = []
    nonlin_vals = []

    for row in df_lin[metric]:
        lin_vals.append(ast.literal_eval(row))
    for row in df_nonlin[metric]:
        nonlin_vals.append(ast.literal_eval(row))

    lin_vals = np.array(lin_vals)
    nonlin_vals = np.array(nonlin_vals)

    avg_lin = np.mean(lin_vals, axis=0)
    avg_nonlin = np.mean(nonlin_vals, axis=0)

    if plot_all:
        for i in range(len(lin_vals)):
            plt.plot(lin_vals[i], color="blue", alpha=0.1)
        for i in range(len(nonlin_vals)):
            plt.plot(nonlin_vals[i], color="red", alpha=0.1)

    plt.plot(avg_lin, label="linear", color="blue")
    plt.plot(avg_nonlin, label="nonlinear", color="red")

    plt.title(f"{metric} over step")
    plt.xlabel("step")
    plt.ylabel(metric)
    plt.grid()
    plt.legend()
    plt.show()




if __name__ == "__main__":
    # Load the data
    df = pd.read_csv("results.csv")
    
    # Plot the validation perplexities
    plot_metric_over_step(df, 'val_pplxs')
    
    # Plot the validation accuracies
    plot_metric_over_step(df, 'val_accs')
    
    # Plot the validation losses
    plot_metric_over_step(df, 'val_losses')
    
    # Plot the training accuracies
    plot_metric_over_step(df, 'train_accs')
    
    # Plot the training losses
    plot_metric_over_step(df, 'train_losses')

