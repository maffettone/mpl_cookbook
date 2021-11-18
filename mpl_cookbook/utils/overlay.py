def add_textbox(ax, text, bbox_props={}):
    props = dict(boxstyle="round", facecolor="C1", alpha=0.2)
    props.update(bbox_props)
    ax.text(
        0.05, 0.95, text, transform=ax.transAxes, verticalalignment="top", bbox=props
    )
