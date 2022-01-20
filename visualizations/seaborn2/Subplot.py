import seaborn as sns
class Subplot:
    def __init__(self,  df_mapping=dict(
            time_col="time",
            object_id_col="object_id",
            f1_col="feature1",
            f2_col="feature2",
            group_col="group_id",
        ),
        
        ):
        self.f1_col = df_mapping["f1_col"]
        self.f2_col = df_mapping["f2_col"]
        self.group_col = df_mapping["group_col"]
        self.object_id_col = df_mapping["object_id_col"]

    def get_style_ts_patch(self,color):
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


    def get_style_rp_ts_patch(self,edgecolor, linewidth):

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


    def _get_sub_plot(self, x, y, z, r=None, **kwargs):

        ax = sns.scatterplot(
            x=x, y=y, **kwargs, marker="s", s=10
        )  # s=0 to hide scatter points
        for i in range(len(x)):
            if r is not None and r.values[i] is True:
                ax.annotate(
                    z.values[i],
                    xy=(x.values[i], y.values[i]),
                    **self.get_style_rp_ts_patch(edgecolor=kwargs["color"], linewidth=2),
                )
                ax.annotate(
                    z.values[i],
                    xy=(x.values[i], y.values[i]),
                    **self.get_style_rp_ts_patch(edgecolor="black", linewidth=1),
                )
            else:
                ax.annotate(
                    z.values[i],
                    xy=(x.values[i], y.values[i]),
                    **self.get_style_ts_patch(color=kwargs["color"]),
                )
        return
    
    def addSubplots(self,g,rp_contained=False):
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
        return g