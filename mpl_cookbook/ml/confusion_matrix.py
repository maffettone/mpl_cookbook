import matplotlib.pyplot as plt
from itertools import product
import numpy as np


class ConfusionMatrixDisplay:
    def __init__(self, *, confusion_matrix, display_labels=None):
        """

        Parameters
        ----------
        confusion_matrix: ndarray
            Array of shape (n_classes, n_classes) holding values of confusion matrix.
        display_labels: ndarray
        """
        self.confusion_matrix = confusion_matrix
        if display_labels is None:
            self.display_labels = np.arange(self.confusion_matrix.shape[0])
        else:
            self.display_labels = display_labels

        self.im_ = None
        self.text_ = None
        self.figure_ = None
        self.ax_ = None

    @classmethod
    def from_predictions(cls, y_true, y_pred, **kwargs):
        from sklearn.metrics import confusion_matrix

        return cls(confusion_matrix=confusion_matrix(y_true, y_pred), **kwargs)

    @classmethod
    def from_txt(cls, path, **kwargs):
        return cls(confusion_matrix=np.loadtxt(path), **kwargs)

    def plot(
        self,
        *,
        include_values=True,
        normalize_values=True,
        cmap="blues",
        xticks_rotation="horizontal",
        values_format=None,
        ax=None,
    ):
        """
        Plot visualization.

        Parameters
        ----------
        include_values: bool, default=True
            Includes values in confusion matrix.
        normalize_values: bool, default=True
            Normalize values as fractions on 0-1.
        cmap: str or matplotlib Colormap, default='coolwarm'
            Colormap recognized by matplotlib.
        xticks_rotation: {'vertical', 'horizontal'} or float, \
                         default='horizontal'
            Rotation of xtick labels.
        values_format: str, default=None
            Format specification for values in confusion matrix. If `None`,
            the format specification is '.2g'.
        ax: matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        Returns
        -------

        display: :class:`ConfusionMatrixDisplay`
        """

        if normalize_values:
            self.confusion_matrix = (
                self.confusion_matrix
                / np.sum(self.confusion_matrix, axis=1)[:, np.newaxis]
            )
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        self.text_ = None

        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)
            if values_format is None:
                values_format = ".2g"

            # print text with appropriate color depending on background
            thresh = (cm.max() + cm.min()) / 2.0
            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min
                self.text_[i, j] = ax.text(
                    j,
                    i,
                    format(cm[i, j], values_format),
                    ha="center",
                    va="center",
                    color=color,
                )

        fig.colorbar(self.im_, ax=ax)
        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=self.display_labels,
            yticklabels=self.display_labels,
            ylabel="True label",
            xlabel="Predicted label",
        )

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self.ax_


class BatchConfusionMatrix(ConfusionMatrixDisplay):
    def __init__(self, *, confusion_matrix, stds, display_labels=None):
        self.stds = stds
        super(BatchConfusionMatrix, self).__init__(
            confusion_matrix=confusion_matrix, display_labels=display_labels
        )

    @classmethod
    def from_predictions(cls, y_true_array, y_pred_array, **kwargs):
        from sklearn.metrics import confusion_matrix

        confusion_matrices = []
        for y_true, y_pred in zip(y_true_array, y_pred_array):
            norm_matrix = confusion_matrix(y_true, y_pred)
            norm_matrix = norm_matrix / np.sum(norm_matrix, axis=1)[:, np.newaxis]
            confusion_matrices.append(norm_matrix)
        confusion_matrices = np.array(confusion_matrices)
        means = np.mean(confusion_matrices, axis=0)
        stds = np.std(confusion_matrices, axis=0)
        return cls(confusion_matrix=means, stds=stds, **kwargs)

    @classmethod
    def from_txt(cls, path, **kwargs):
        raise NotImplementedError

    def plot(self, **kwargs):
        kwargs.pop("normalize_values", None)
        kwargs.pop("include_values", None)
        super().plot(normalize_values=False, include_values=True, **kwargs)
        for i, j in product(
            range(self.confusion_matrix.shape[0]), range(self.confusion_matrix.shape[0])
        ):
            self.text_[i, j].set_text(
                f"{self.confusion_matrix[i,j]:.2f} +/- {self.stds[i,j]:.2f}"
            )

        return self.ax_
