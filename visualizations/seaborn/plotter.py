import pandas as pd
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.pyplot as plt



def get_style_ts_patch(color):
    style_ts_patch = dict(
        fontsize=10,
        xytext=(-2, -1.5),
        textcoords="offset points",
        bbox=dict(boxstyle="square", alpha=0.1, color=color),
        va="top",
        ha="right",
        alpha=0.4,
    )
    return style_ts_patch


def get_style_rp_ts_patch(edgecolor, linewidth):

    style_rp_ts_patch = dict(
        fontsize=10,
        xytext=(-2, -1.5),
        textcoords="offset points",
        bbox=dict(
            boxstyle="square",
            facecolor="none",
            linestyle="--",
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=1,
        ),
        va="top",
        ha="right",
        alpha=0,
    )
    return style_rp_ts_patch


def get_style_rp_legend_patch(edgecolor, linewidth):
    style_rp_legend_patch = dict(
        linestyle="--",
        facecolor=None,
        fill=False,
        alpha=1,
        linewidth=linewidth,
        edgecolor=edgecolor,
    )
    return style_rp_legend_patch


class Plotter:
    def __init__(
        self,
        df,
        df_mapping=dict(
            time_col="time",
            object_id_col="object_id",
            f1_col="feature1",
            f2_col="feature2",
            group_col="group_id",
        ),
        plot_settings=dict(col_wrap=3, bbox_to_anchor=(-1.2, -0.4)),
    ):
        self.df = df
        self.time_col = df_mapping["time_col"]
        self.f1_col = df_mapping["f1_col"]
        self.f2_col = df_mapping["f2_col"]
        self.group_col = df_mapping["group_col"]
        self.object_id_col = df_mapping["object_id_col"]
        self.col_wrap = plot_settings["col_wrap"]
        self.bbox_to_anchor = plot_settings["bbox_to_anchor"]

    def extend_df_by_cluster_path_group(self):
        all_cluster_paths = (
            self.df.groupby(["object_id"])
            .cluster_id.apply(tuple)
            .reset_index(name="cluster_path")
        )

        cluster_path_groups = (
            all_cluster_paths.drop(columns=["object_id"])
            .drop_duplicates("cluster_path")
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "group_id"})
        )

        group_assignments = cluster_path_groups.merge(
            all_cluster_paths, on="cluster_path", how="left"
        ).drop(columns=["cluster_path"])

        df_cluster_path_group_extended = self.df.merge(
            group_assignments, on="object_id", how="left"
        )
        return df_cluster_path_group_extended

    def _get_sub_plot(self, x, y, z, r=None, **kwargs):

        ax = sns.scatterplot(
            x=x, y=y, **kwargs, marker="s", s=10
        )  # s=0 to hide scatter points
        for i in range(len(x)):
            if r is not None and r.values[i] is True:
                ax.annotate(
                    z.values[i],
                    xy=(x.values[i], y.values[i]),
                    **get_style_rp_ts_patch(edgecolor=kwargs["color"], linewidth=2),
                )
                ax.annotate(
                    z.values[i],
                    xy=(x.values[i], y.values[i]),
                    **get_style_rp_ts_patch(edgecolor="black", linewidth=1),
                )
            else:
                ax.annotate(
                    z.values[i],
                    xy=(x.values[i], y.values[i]),
                    **get_style_ts_patch(color=kwargs["color"]),
                )
        return

    def _get_legend_items(self, cluster_groups, rp=False):

        groups_labels = list(map(lambda x: f"group {x}", cluster_groups))
        rp_labels = list(map(lambda x: f"rp ts {x}", cluster_groups))

        colors = sns.color_palette("Set1").as_hex()
        handles_groups = [patches.Patch(color=col, alpha=0.3) for col in (colors)]
        handles = []
        labels = []
        if rp:
            handles_rp = []
            for col in colors:
                first_patch = patches.Patch(
                    **get_style_rp_legend_patch(edgecolor=col, linewidth=2)
                )
                second_patch = patches.Patch(
                    **get_style_rp_legend_patch(edgecolor="black", linewidth=1)
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

    def generate_fig(self):

        df_cluster_groups = self.extend_df_by_cluster_path_group()

        sns.set(style="darkgrid")
        g = sns.FacetGrid(
            col_wrap=self.col_wrap,
            data=df_cluster_groups,
            col=self.time_col,
            palette="Set1",
            hue="group_id",
        )
        rp_contained = True if "representative" in self.df.columns else False
        if rp_contained:
            g.map(
                self._get_sub_plot,
                self.f1_col,
                self.f2_col,
                self.object_id_col,
                "representative",
            )
        else:
            g.map(self._get_sub_plot, self.f1_col, self.f2_col, self.object_id_col)

        cluster_groups = df_cluster_groups["group_id"].unique()
        items = self._get_legend_items(cluster_groups, rp=rp_contained)

        plt.legend(
            items["handles"],
            items["labels"],
            loc="upper left",
            ncol=(len(cluster_groups)),
            bbox_to_anchor=self.bbox_to_anchor,
            title="Legend",
        )

        return g

    def add_representatives(self, df):
        df["representative"] = True
        merged_df = pd.concat([self.df, df]).reset_index(drop=True)
        self.df = merged_df
