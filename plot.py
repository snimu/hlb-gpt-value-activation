"""Plot the results"""

import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric_over_step(df, metric, plot_all: bool = False, show: bool = True):
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

    step_col = "train_steps" if "train" in metric else "val_steps"
    steps = np.array(ast.literal_eval(df_lin[step_col].iloc[0]))

    if plot_all:
        for i in range(len(lin_vals)):
            plt.plot(steps, lin_vals[i], color="blue", alpha=0.1)
        for i in range(len(nonlin_vals)):
            plt.plot(steps, nonlin_vals[i], color="orange", alpha=0.1)

    plt.plot(steps, avg_lin, label="linear", color="blue", linewidth=2)
    plt.plot(steps, avg_nonlin, label="nonlinear", color="orange", linewidth=2)

    plt.title(f"{metric}: linear vs nonlinear value, averaged over {len(lin_vals)} runs")
    plt.xlabel("step")
    plt.ylabel(metric)
    plt.grid()
    plt.legend()
    plt.tight_layout()

    if show:
        plt.show()
    else:
        plt.savefig(f"results/{metric}.png", dpi=300)
    
    plt.cla()
    plt.clf()
    plt.close()


if __name__ == "__main__":
    # Load the data
    df = pd.read_csv("results/results.csv")
    show = False
    plot_all = True
    
    # Plot the validation perplexities
    plot_metric_over_step(df, 'val_pplxs', plot_all=plot_all, show=show)
    
    # Plot the validation accuracies
    plot_metric_over_step(df, 'val_accs', plot_all=plot_all, show=show)
    
    # Plot the validation losses
    plot_metric_over_step(df, 'val_losses', plot_all=plot_all, show=show)
    
    # Plot the training accuracies
    plot_metric_over_step(df, 'train_accs', plot_all=plot_all, show=show)
    
    # Plot the training losses
    plot_metric_over_step(df, 'train_losses', plot_all=plot_all, show=show)

