## General Usage

In order to use the transition-based outlier detection algorithm DACT on your clustered data, first the object has to be initialized:

```python
import pandas
from ots_eval.outlier_detection.dact import DACT

data = pandas.DataFrame(data, columns=['object_id', 'time', 'cluster_id'])
detector = DACT(data)
```

The names of the columns in the DataFrames are not relevant but the order of them. The DataFrame may contain further columns but only the first three are considered.

The clusters can then be calculated by calling

```python
outlier_result = detector.calc_outlier_degree()
clusters, outlier_result = detector.mark_outliers(tau=0.3)
```

With the threshold parameter `tau` the function `mark_outliers` computes all outliers with `outlier_score > tau` and returns the data _DataFrame_ with an additional column _&#39;outlier&#39;_ indicating, if a data point is an outlier and which type of outlier it is, and the outlier result, which is a _pandas.DataFrame_ with columns _&#39;object\_id&#39;, &#39;start\_time&#39;, &#39;end\_time&#39;, &#39;cluster\_end\_time&#39;, &#39;rating&#39;, &#39;distance&#39;_ and _&#39;outlier&#39;_. The outlier types are:

*   `-1`: transition-based outlier
*   `-2`: intuitive outlier

With

```python
clusters, outlier_result = detector.get_outliers(tau=0.3)
```

the outliers are calculated immediately.

#### Specified Outlier Amount
If it is wanted to detect a certain amount of the data as outliers `mark_outliers` can be used with the parameter `percent`:

```python
outlier_result = detector.calc_outlier_degree()
clusters, outlier_result = detector.mark_outliers(percent=0.1)
```

where `0.0 <= percent <= 1.0` describes the relative amount of the data that should be considered outliers.

#### Statistic Outliers
With `calc_statistic_cluster_outliers`outliers with respect to the standard deviation of the respective cluster can be calculated:

```python
clusters, outlier_result = detector.calc_statistic_cluster_outliers(factor=3.0)
```

The `factor` describes that all subsequences with a greater deviation than `3.0*stddev` are considered outliers.