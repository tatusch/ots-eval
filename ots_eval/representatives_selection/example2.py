from ots_eval.clustering.cots import COTS
from representatives import Representatives
from visualizations.plotly.plotter_3d import Plotter3d

import pandas as pd



def get_data():
    folder_path = "../../data/"
    csv_name = "pub1_generated_datasetA.csv"
    df = pd.read_csv(folder_path + csv_name)
    return df


def get_clustering(df):
    cots = COTS(df)
    clusters = cots.get_clusters_df(min_cf=0.2, sw=3)
    return clusters


def get_representatives(df):
    rp = Representatives()
    representatives = rp.get_representatives(df)
    return representatives


def plot_result(df, representatives):
    pl = Plotter3d(df)
    pl.add_representatives(representatives)
    pl.generate_fig()
    fig = pl.fig_3d
    fig.show()


if __name__ == "__main__":
    data = get_data()
    clustering = get_clustering(data)
    clustering.rename(columns={"cluster": "cluster_id"}, inplace=True)
    rp = get_representatives(clustering)
    plot_result(clustering, rp)
