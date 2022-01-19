import pandas as pd
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
from ots_eval.representatives_selection.representatives import Representatives

# from plotter import Plotter


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
    ):
        self.df = df
        self.time_col = df_mapping["time_col"]
        self.f1_col = df_mapping["f1_col"]
        self.f2_col = df_mapping["f2_col"]
        self.group_col = df_mapping["group_col"]
        self.object_id_col = df_mapping["object_id_col"]

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
        ax = sns.scatterplot(x=x, y=y, **kwargs, s=0)
        for i in range(len(x)):
            if r is not None and r.values[i] is True:
                ax.annotate(
                    z.values[i],
                    xy=(x.values[i], y.values[i]),
                    fontsize=15,
                    xytext=(0, 0),
                    textcoords="offset points",
                    bbox=dict(
                        boxstyle="round",
                        facecolor="none",
                        linestyle="--",
                        edgecolor=kwargs["color"],
                        alpha=1,
                    ),
                    va="center",
                    ha="center",
                    weight="bold",
                    alpha=0,
                )
            else:
                ax.annotate(
                    z.values[i],
                    xy=(x.values[i], y.values[i]),
                    fontsize=15,
                    xytext=(0, 0),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", alpha=0.3, color=kwargs["color"]),
                    va="center",
                    ha="center",
                    weight="bold",
                    alpha=1,
                )
        return

    def _get_legend_handles(self, cluster_groups,rp=False):

        groups_labels = list(map(lambda x: f"group {x}", cluster_groups))
        rp_labels = list(map(lambda x: f"rp ts {x}", cluster_groups))

        colors = sns.color_palette("Set1").as_hex()
        handles_groups = [
            patches.Patch(color=col, alpha=0.3, label=lab)
            for col, lab in zip(colors, groups_labels)
        ]
        handles=[]
        if rp:
            handles_rp = [
                patches.Patch(
                    edgecolor=col,
                    linestyle="--",
                    facecolor=None,
                    fill=False,
                    alpha=1,
                    label=lab,
                )
                for col, lab in zip(colors, rp_labels)
            ]

            handles = [None] * (len(handles_groups) + len(handles_rp))
            handles[::2] = handles_groups
            handles[1::2] = handles_rp
        else:
            handles=handles_groups
        return handles

    def generate_fig(self):
        df_cluster_groups = self.extend_df_by_cluster_path_group()
        cluster_groups = df_cluster_groups["group_id"].unique()
        rp_contained=True if "representative" in self.df.columns else False
        sns.set(style="darkgrid")
        g = sns.FacetGrid(
            data=df_cluster_groups, col=self.time_col, palette="Set1", hue="group_id"
        )

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

        handles = self._get_legend_handles(cluster_groups,rp=rp_contained)
        g.add_legend(
            handles=handles,
            title="Legend",
            loc="upper left",
            ncol=(len(cluster_groups)),
            bbox_to_anchor=(0.23, 0),
        )
        return g

    def add_representatives(self, df):
        df["representative"] = True
        merged_df = pd.concat([self.df, df]).reset_index(drop=True)
        self.df = merged_df


def get_data():
    test_data = [
        [1, 1, 1, 1 / 3, 1 / 6],
        [2, 1, 1, 2 / 3, 1 / 6],
        [3, 1, 1, 1 / 3, 2 / 6],
        [4, 1, 2, 2 / 3, 4 / 6],
        [5, 1, 2, 3 / 3, 4 / 6],
        [6, 1, 2, 2 / 3, 5 / 6],
        [7, 1, 7, 0.5, 0.5],
        [1, 2, 3, 2 / 3, 1 / 6],
        [2, 2, 3, 3 / 3, 1 / 6],
        [3, 2, 3, 2 / 3, 2 / 6],
        [4, 2, 4, 2 / 3, 5 / 6],
        [5, 2, 4, 3 / 3, 5 / 6],
        [6, 2, 4, 2 / 3, 6 / 6],
        [7, 2, 7, 0.5, 0.5],
        [1, 3, 5, 2 / 3, 1 / 6],
        [2, 3, 5, 2 / 3, 2 / 6],
        [3, 3, 5, 1 / 3, 1 / 6],
        [4, 3, 6, 2 / 3, 5 / 6],
        [5, 3, 6, 3 / 3, 4 / 6],
        [6, 3, 6, 1 / 3, 6 / 6],
        [7, 3, 7, 0.5, 0.5],
    ]

    data = pd.DataFrame(
        test_data, columns=["object_id", "time", "cluster_id", "feature1", "feature2"]
    )
    return data


data = get_data()


rp = Representatives()
extended_test = rp.extend_df_by_cluster_path_group(data)
print("extended df:")
print(extended_test)
print()
rp_data = rp.get_representatives(data)
print("only rps:")
print(rp_data)
plotter = Plotter(data)
plotter.add_representatives(rp_data)
fig_ts_only = plotter.generate_fig()

fig_ts_only.savefig("testn.png")
