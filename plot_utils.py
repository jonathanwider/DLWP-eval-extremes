# https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
def add_headers(
    fig,
    row_headers=None,
    col_headers=None,
    cbar_headers=None,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    **text_kwargs,
):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()
    if cbar_headers is not None:
        axes = [a for a in axes if a.get_ylabel() not in cbar_headers]
    else:
        axes = list(axes)
    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )
