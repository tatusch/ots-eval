import seaborn as sns
import matplotlib.patches as patches


class Legend:
    def __init__(
        self,
        df_cluster_groups,
        plot_settings=dict(col_wrap=3, bbox_to_anchor=(-1.2, -0.4)),
    ):
        self.df_cluster_groups = df_cluster_groups
        self.bbox_to_anchor = plot_settings["bbox_to_anchor"]

    def _get_style_rp_legend_patch(self, edgecolor, linewidth):
        style_rp_legend_patch = dict(
            linestyle="--",
            facecolor=None,
            fill=False,
            alpha=1,
            linewidth=linewidth,
            edgecolor=edgecolor,
        )
        return style_rp_legend_patch

    def _get_style_ts_legend_patch(self, color):
        style_ts_legend_patch = dict(alpha=0.3, color=color)
        return style_ts_legend_patch

    def _get_legend_items(self, cluster_groups, rp_contained=False):
        groups_labels = list(map(lambda x: f"group {x}", cluster_groups))
        rp_labels = list(map(lambda x: f"rp ts {x}", cluster_groups))

        colors = sns.color_palette("Set1").as_hex()
        handles_groups = []
        for col in colors:
            p = patches.Patch(**self._get_style_ts_legend_patch(color=col))
            handles_groups.append(p)

        handles = []
        labels = []
        if rp_contained:
            handles_rp = []
            for col in colors:
                first_patch = patches.Patch(
                    **self._get_style_rp_legend_patch(edgecolor=col, linewidth=2)
                )
                second_patch = patches.Patch(
                    **self._get_style_rp_legend_patch(edgecolor="black", linewidth=1)
                )
                handles_rp.append((first_patch, second_patch))

            handles = [None] * (len(handles_groups) + len(handles_rp))
            handles[::2] = handles_groups
            handles[1::2] = handles_rp
            labels = [None] * (len(groups_labels) + len(rp_labels))
            labels[::2] = groups_labels
            labels[1::2] = rp_labels
        else:
            handles = handles_groups
            labels = groups_labels
        return {"handles": handles, "labels": labels}

    def addLegend(self, plt, rp_contained=False):
        cluster_groups = self.df_cluster_groups["group_id"].unique()
        items = self._get_legend_items(cluster_groups, rp_contained=rp_contained)
        plt.legend(
            items["handles"],
            items["labels"],
            loc="upper left",
            ncol=(len(cluster_groups)),
            bbox_to_anchor=self.bbox_to_anchor,
            title="Legend",
        )
