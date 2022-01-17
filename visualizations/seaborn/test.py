import pandas as pd


from ots_eval.representatives_selection.representatives import Representatives
from plotter import Plotter
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



data=get_data()


rp=Representatives()
rp_data=rp.get_representatives(data)
plotter=Plotter(data)
plotter.add_representatives(rp_data)
fig_ts_only=plotter.generate_fig()
fig_ts_only.savefig('testn.png')


