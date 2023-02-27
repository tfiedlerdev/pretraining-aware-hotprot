import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab as pl
import seaborn as sns
import os


def plot_advanced_scatter(predictions, actuals, outPath):
    x = predictions
    y = actuals
    slope, intercept = np.polyfit(x, y, 1)  # linear model adjustment

    y_model = np.polyval([slope, intercept], x)  # modeling...

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    n = len(x)  # number of samples
    m = 2  # number of parameters
    dof = n - m  # degrees of freedom
    t = stats.t.ppf(0.975, dof)  # Students statistic of interval confidence

    residual = y - y_model

    std_error = (np.sum(residual**2) / dof) ** 0.5  # Standard deviation of the error

    # calculating the r2
    # https://www.statisticshowto.com/probability-and-statistics/coefficient-of-determination-r-squared/
    # Pearson's correlation coefficient
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = (np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2)) ** 0.5
    correlation_coef = numerator / denominator
    r2 = correlation_coef**2

    # mean squared error
    MSE = 1 / n * np.sum((y - y_model) ** 2)

    # to plot the adjusted model
    x_line = np.linspace(np.min(x), np.max(x), 100)
    y_line = np.polyval([slope, intercept], x_line)

    # confidence interval
    ci = (
        t
        * std_error
        * (1 / n + (x_line - x_mean) ** 2 / np.sum((x - x_mean) ** 2)) ** 0.5
    )
    # predicting interval
    pi = (
        t
        * std_error
        * (1 + 1 / n + (x_line - x_mean) ** 2 / np.sum((x - x_mean) ** 2)) ** 0.5
    )

    # ------------------------ Ploting
    plt.rcParams.update({"font.size": 14})
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    ax.plot(x, y, "o", color="royalblue", alpha=0.1)
    ax.plot(x_line, y_line, color="royalblue")
    ax.fill_between(
        x_line,
        y_line + pi,
        y_line - pi,
        color="lightcyan",
        label="95% prediction interval",
    )
    ax.fill_between(
        x_line,
        y_line + ci,
        y_line - ci,
        color="skyblue",
        label="95% confidence interval",
    )

    ax.set_xlabel("predictions")
    ax.set_ylabel("actuals")

    # rounding and position must be changed for each case and preference
    a = str(np.round(intercept))
    b = str(np.round(slope, 2))
    r2s = str(np.round(r2, 2))
    MSEs = str(np.round(MSE))

    ax.text(45, 110, "y = " + a + " + " + b + " x")
    ax.text(45, 100, "$r^2$ = " + r2s + "     MSE = " + MSEs)

    plt.legend(bbox_to_anchor=(1, 0.25), fontsize=12)
    plt.savefig(outPath)


def plot_predictions(
    out_base_label: str,
    plot_title: str,
    preds: "list[float]",
    actuals: "list[float]",
    output_dir="results",
):
    pl.scatter(preds, actuals, alpha=0.15)
    fileName = f"{out_base_label}.png"
    os.makedirs(output_dir, exist_ok=True)
    plotPath = os.path.join(output_dir, f"{fileName}")

    pl.title(plot_title)
    pl.xlabel("Predictions")
    pl.ylabel("Labels")
    pl.savefig(plotPath)
    fig, ax = plt.subplots()

    ax = sns.regplot(x=preds, y=actuals, label=plot_title, ax=ax)

    ax.set_xlabel("Predictions")
    ax.set_ylabel("Actuals")
    seabornPath = os.path.join(output_dir, f"seaborn_{fileName}")
    fig.savefig(seabornPath)
    advancedPath = os.path.join(output_dir, f"advanced_{fileName}")
    plot_advanced_scatter(preds, actuals, advancedPath)

    plt.clf()
    plt.hist(
        np.subtract(preds, actuals),
        label="Frequency of differences of predicted vs actual melting points (°C)",
        bins=100,
    )
    pl.xlabel("Difference predicted - actual melting point")
    pl.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "hist.png"))
    print(
        f"Saved predictions as \n- scatter plot {plotPath}\n- seaborn scatter {seabornPath}\n- advanced scatter {advancedPath}"
    )
