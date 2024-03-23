"""Plot the results"""

import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def close_plot():
    plt.cla()
    plt.clf()
    plt.close()


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
    
    close_plot()


def plot_metrics(df: pd.DataFrame, show: bool = True, plot_all: bool = True):
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


def plot_spread_of_tokens_seen(df: pd.DataFrame, train: bool = False):
    """An eventplot of the tokens seen over the different run_number between linear and non-linear value.
    Where the tokens seen is the final entry in tokens_seen_train or tokens_seen_val, depending on the train argument.
    """
    df_lin = df[df["linear_value"]]
    df_nonlin = df[~df["linear_value"]]

    tokens_lin = []
    tokens_nonlin = []

    for row in df_lin["tokens_seen_train" if train else "tokens_seen_val"]:
        tokens_lin.append(ast.literal_eval(row)[-1])
    for row in df_nonlin["tokens_seen_train" if train else "tokens_seen_val"]:
        tokens_nonlin.append(ast.literal_eval(row)[-1])

    tokens_lin = np.array(tokens_lin)
    tokens_nonlin = np.array(tokens_nonlin)
    all_tokens = np.vstack([tokens_lin, tokens_nonlin])

    fig, ax = plt.subplots()
    ax.eventplot(all_tokens, colors=["blue", "orange"], lineoffsets=[0, 1], linelengths=0.8, orientation="vertical")
    ax.legend(["linear", "nonlinear"])
    ax.set_xticklabels([])

    plt.title("Tokens seen: linear (left) vs nonlinear (right) value")
    plt.ylabel("tokens seen")
    plt.grid()
    plt.tight_layout()
    plt.show()
    close_plot()


def plot_final_metric_over_final_time_taken(df: pd.DataFrame, metric: str):
    df_lin = df[df["linear_value"]]
    df_nonlin = df[~df["linear_value"]]

    final_metric_lin = df_lin[metric].apply(lambda x: ast.literal_eval(x)[-1])
    final_metric_nonlin = df_nonlin[metric].apply(lambda x: ast.literal_eval(x)[-1])
    final_times_lin = df_lin["cumulative_time_taken"].apply(lambda x: ast.literal_eval(x)[-1])
    final_times_nonlin = df_nonlin["cumulative_time_taken"].apply(lambda x: ast.literal_eval(x)[-1])

    plt.scatter(final_times_lin, final_metric_lin, color="blue", label="linear")
    plt.scatter(final_times_nonlin, final_metric_nonlin, color="orange", label="nonlinear")

    plt.title(f"Final {metric} over final time taken: linear vs nonlinear value")
    plt.xlabel("final time taken (seconds)")
    plt.ylabel(metric)

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    close_plot()


def main():
    df = pd.read_csv("results/results_25_tries_1000_steps_40Mparams.csv")
    plot_metrics(df)
    plot_spread_of_tokens_seen(df)
    plot_final_metric_over_final_time_taken(df, "val_pplxs")


if __name__ == "__main__":
    main()
