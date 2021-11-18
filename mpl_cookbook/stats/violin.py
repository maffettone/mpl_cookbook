import matplotlib.pyplot as plt
import numpy as np


def adjacent_values(vals, q1, q3):
    """Get adjacent values for upper and lower quartiles"""
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, np.max(vals))

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, np.min(vals), q1)
    return lower_adjacent_value, upper_adjacent_value


def make_categorical_comparison_plot(
    value_dict,
    *,
    colors_dict=None,
    labels_dict=None,
    ax=None,
    line_overlay=False,
    ci_overlay=False,
    **kwargs,
):
    """
    Make a comparison plot of colored groups of violin-plots over an
    shared x-axis labeling.

    For example, looking at men/women (color) car insurance costs (y-axis)
    over age bracket (x-axis label).
    Parameters
    ----------
    value_dict: dict
        Form {(x-axis_group, color_group):[value list]}
        Dictionary of lists of values to assemble box_plots
    colors_dict: dict
        Form {color_group: color}
        Where color value is a valid matplotlib color.
        Defaults to matplotlib color cycle.
    labels_dict: dict
        Form {x-axis_group, "text_label"}
    ax: mpl.axis
        Optional axis, if None given, a figure will be created
    line_overlay: bool
        Overlay line plot of the medians for the boxes
    ci_overlay: bool
        Overlay shaded area of confidence interval.
    kwargs: dict
        Keyword arguments for plt.figure()
    Returns
    -------

    """

    parts = None
    if ax is None:
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot()

    # Assemble axis group list, retaining order, and not using set efficiencies.
    axis_group_list = []
    color_group_list = []
    for key in value_dict.keys():
        if key[0] not in axis_group_list:
            axis_group_list.append(key[0])
        if key[1] not in color_group_list:
            color_group_list.append(key[1])

    n_axis_group = len(axis_group_list)
    n_color_group = len(color_group_list)
    x_ticks = np.arange(1, n_axis_group + 1)

    # Default to using axis group as labels and color cycle over set of color_groups
    if labels_dict is None:
        labels_dict = {k: f"{k}" for k in axis_group_list}
    if colors_dict is None:
        colors_dict = {k: f"C{i}" for i, k in enumerate(color_group_list)}

        offset = 0
    legend_txt = []
    legend_artists = []
    for color_group in color_group_list:
        legend_txt.append(color_group)
        data = []
        for axis_group in axis_group_list:
            data.append(value_dict[(axis_group, color_group)])

        parts = ax.violinplot(
            data,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.5,
            positions=x_ticks + offset,
            widths=1 / (n_color_group + 1),
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(colors_dict[color_group])
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)
        quartile1, medians, quartile3 = np.array(
            [np.percentile(arr, [25, 50, 75]) for arr in data]
        ).T
        whiskers = np.array(
            [
                adjacent_values(array, q1, q3)
                for array, q1, q3 in zip(data, quartile1, quartile3)
            ]
        )
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
        inds = x_ticks + offset
        ax.scatter(inds, medians, marker="o", color="white", s=10, zorder=3)
        ax.vlines(inds, quartile1, quartile3, color="k", linestyle="-", lw=3)
        ax.vlines(inds, whiskers_min, whiskers_max, color="k", linestyle="-", lw=1)
        legend_artists.append(parts["bodies"][0])

    center_points = x_ticks + (n_color_group - 1) / (
        2 * (n_color_group + 1)
    )  # Works at least for n=2,3,4
    ax.set_xticks(center_points)
    ax.set_xticklabels(
        [labels_dict[axis_group] for axis_group in axis_group_list], rotation=0
    )
    ax.legend(legend_artists, legend_txt, loc="lower right")

    if line_overlay or ci_overlay:
        for color_group in color_group_list:
            data = []
            for axis_group in axis_group_list:
                data.append(value_dict[(axis_group, color_group)])
            data = np.array(data)
            means = np.mean(data, axis=-1)
            stds = np.std(data, axis=-1)
            if line_overlay:
                ax.plot(
                    center_points,
                    means,
                    color=colors_dict[color_group],
                    linestyle=(0, (5, 7)),
                    zorder=-1,
                )
            if ci_overlay:
                ax.fill_between(
                    center_points,
                    means - stds,
                    means + stds,
                    color="grey",
                    alpha=0.2,
                    zorder=-1,
                )

    return ax, parts
