import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# function to display 

# function to create a boxplot of a feature based on a categorical target
def sns_box_plot_by_target(
    df: pd.DataFrame,
    col: str,
    target: str = "target",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple[int, int] = (6, 4),
    palette: str | None = None) -> None:
    """
    Plot a Seaborn boxplot of a numeric column grouped by a categorical target variable.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    col : str
        Column name of the numeric variable to plot.
    target : str, default="target"
        Column name of the categorical target variable.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    figsize : tuple[int, int], default=(6, 4)
        Size of the figure in inches.
    palette : str or None, optional
        Color palette for the boxplot.
    """
    # initialize plot
    plt.figure(figsize=figsize)

    # create plot
    ax = sns.boxplot(x=target, y=col, data=df, palette=palette)

    # assign labels
    ax.set_title(title if title else f"{col} by {target}")
    ax.set_xlabel(xlabel if xlabel else target)
    ax.set_ylabel(ylabel if ylabel else col)

    # show plot
    sns.despine()
    plt.tight_layout()
    plt.show()

# function to plot a histogram for continuous variables
def sns_hist_plot(
    df: pd.DataFrame,
    col: str,
    bins: int = 30,
    hue: str | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Frequency",
    figsize: tuple[int, int] = (6, 4),
    kde: bool = False,
    palette: str | None = None,
    alpha: float = 0.7) -> None:
    """
    Plot a Seaborn histogram for a numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    col : str
        Column name of the numeric variable to plot.
    bins : int, default=30
        Number of bins for the histogram.
    hue : str or None, optional
        Column name for color grouping (categorical variable).
    title : str, optional
        Title of the plot. Defaults to 'Histogram of <col>'.
    xlabel : str, optional
        Label for the x-axis. Defaults to col name.
    ylabel : str, default="Frequency"
        Label for the y-axis.
    figsize : tuple[int, int], default=(6, 4)
        Size of the figure in inches.
    kde : bool, default=False
        Whether to overlay a kernel density estimate (KDE).
    palette : str or None, optional
        Color palette for the histogram if hue is provided.
    alpha : float, default=0.7
        Transparency level of the bars (0–1).
    """
    # initialize plot
    plt.figure(figsize=figsize)

    # create plot
    ax = sns.histplot(
        data=df,
        x=col,
        bins=bins,
        hue=hue,
        kde=kde,
        palette=palette,
        alpha=alpha
    )

    # assign labels
    ax.set_title(title if title else f"Histogram of {col}")
    ax.set_xlabel(xlabel if xlabel else col)
    ax.set_ylabel(ylabel)

    # show plot
    sns.despine()
    plt.tight_layout()
    plt.show()
