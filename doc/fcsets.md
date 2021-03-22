## General Usage

When using FCSETS as stability measure for fuzzy over-time clusterings, first, FCSETS has to be initialized:

```python
import pandas
import numpy
from stability_evaluation.fcsets import FCSETS

# test_data for 5 time series with 4 timestamps and 2 clusters per year
data = {
    1: numpy.array([[0.999, 0.9905, 0.989, 0.002, 0.028],
                    [0.001, 0.0005, 0.011, 0.998, 0.972]]),
    2: numpy.array([[0.999, 0.9905, 0.989, 0.002, 0.028],
                    [0.001, 0.0005, 0.011, 0.998, 0.972]]),
    3: numpy.array([[0.999, 0.9905, 0.989, 0.002, 0.028],
                    [0.001, 0.0005, 0.011, 0.998, 0.972]]),
    4: numpy.array([[0.999, 0.9905, 0.989, 0.002, 0.028],
                    [0.001, 0.0005, 0.011, 0.998, 0.972]])
}
rater = FCSETS(data)
```
FCSETS requires a `dictionary` with the form `{<timestamp>: <membership_martix>}` with membership_matrix being a 2-dim numpy.array of shape `(num_clusters, num_time_series)`

Now, the clustered data set can be evaluated with FCSETS. 

```python
clustering_score = rater.rate_clustering(start_time=None, end_time=None)
```

where _start\_time_ and _end\_time_ indicate the time intervall which should be considered. If _start\_time_ and _end\_time_ are _None_, the first and last timestamp are considered as boundary, respectively.
