import matplotlib.pyplot as plt
import numpy as np


def make_metrics_plot(
    metrics_dict, keys, *, colors=None, axis_label=None, ax=None, **kwargs
):
    """
    Makes line plot of metrics
    Parameters
    ----------
    metrics_dict: dict
        Dictionary of lists or arrays to plot. Generally the metrics over epochs.
    keys: list[str]
        Length 2 or 3 list of keys for metrics dict to plot (train, validation, [test])
    colors: list
        Length 2 or 3 list of valid matplotlib colors
    axis_label: str
        Text label, defaults to first key
    kwargs: dict
        Keyword arguments for subplots

    Returns
    -------
    ax

    """

    with plt.style.context("bmh"):
        if ax is None:
            fig, ax = plt.subplots(1, 1, **kwargs)

        ax.set_ylabel(axis_label)
        ax.set_xlabel("Epochs")

        labels = ["Train", "Val", "Test"][: len(keys)]
        if colors is None:
            colors = ["C0", "C1", "C2"][: len(keys)]

        for label, color, key in zip(labels, colors, keys):
            ax.plot(
                np.arange(len(metrics_dict[key])),
                metrics_dict[key],
                label=label,
                color=color,
            )
        ax.legend()

    return ax


def make_torch_regression_scatter(
    model, train_dataset, val_dataset=None, *, ax=None, n_points=1000, batch_size=1000, **kwargs
):
    """
    Takes datasets, makes predictions, plots random subset in scatter

    Parameters
    ----------
    model: Callable
        Torch model to be called on data batch
    train_dataset: Dataset
    val_dataset: Dataset
        Optional
    ax: axis
        Optional
    n_points: int
        Number of scatter points to include
    kwargs: dict
        Keyword arguments for subplots

    Returns
    -------
    ax

    """
    from torch.utils.data import DataLoader

    with plt.style.context("bmh"):
        if ax is None:
            fig, ax = plt.subplots(1, 1, **kwargs)

        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        y_hats = []
        ys = []
        for i, (x, y) in enumerate(loader):
            if i * batch_size > n_points:
                break
            y_hats.append(model(x).cpu().detach().numpy())
            ys.append(y.cpu().detach().numpy())
        train_y = np.concatenate(ys)
        train_y_hat = np.concatenate(y_hats)

        ax.set_title("Model predictions", fontsize=12)
        ax.scatter(train_y, train_y_hat, label="Train", s=1)
        bounds = (
            min(np.min(train_y), np.min(train_y_hat)),
            max(np.max(train_y), np.max(train_y_hat)),
        )
        ax.plot(bounds, bounds, "k")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_ylim(*bounds)
        ax.set_xlim(*bounds)

        if val_dataset:
            loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
            y_hats = []
            ys = []
            for i, (x, y) in enumerate(loader):
                if i * batch_size > n_points // 4:
                    break
                y_hats.append(model(x).cpu().detach().numpy())
                ys.append(y.cpu().detach().numpy())
            val_y = np.concatenate(ys)
            val_y_hat = np.concatenate(y_hats)
            ax.scatter(val_y, val_y_hat, label="Val", s=1)

    return ax
