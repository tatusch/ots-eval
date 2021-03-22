## General Usage

In order to use the transition-based outlier detection algorithm DOOTS on your clustered data, first the object has to be initialized:

```python
import pandas
from ots_eval.outlier_detection.doots import DOOTS

data = pandas.DataFrame(data, columns=['object_id', 'time', 'cluster_id', 'feature1', 'feature2'])
detector = DOOTS(data, weighting=False, jaccard=False)
```

Explanation of the parameter:

<figure class="table"><table><thead><tr><th>Parameter</th><th><br data-cke-filler="true"></th><th>Default</th><th>Datatype</th><th>Description</th></tr></thead><tbody><tr><td><code>data</code></td><td>-</td><td>-</td><td><i>pandas.DataFrame</i></td><td>with first column being the objectID, second being the timestamp, third being the clusterID and following columns being the features</td></tr><tr><td><code>jaccard</code></td><td>optional</td><td><i>False</i></td><td><i>boolean</i></td><td>indicating if jaccard index should be used</td></tr><tr><td><code>weighting</code></td><td>optional</td><td><i>False</i></td><td><i>boolean</i></td><td>indicating if more distant past should be weighted lower than nearer past</td></tr></tbody></table></figure>

The names of the columns in the DataFrames are not relevant but the order of them.

The outliers can then be calculated by calling

```python
outlier_result = detector.calc_outlier_degree()
clusters, outlier_result = detector.mark_outliers(tau=0.5)
```

The function `calc_outlier_degree` computes the degree of being an outlier for every subsequence. With `mark_outliers` and the threshold parameter `tau` all outliers are marked. The function returns the data _DataFrame_ with an additional column _&#39;outlier&#39;_ indicating, if a data point is an outlier and which type of outlier it is, and the outlier result, which is a _pandas.DataFrame_ with columns _&#39;object\_id&#39;, &#39;start\_time&#39;, &#39;end\_time&#39;, &#39;cluster\_end\_time&#39;, &#39;rating&#39;, &#39;distance&#39;_ and _&#39;outlier&#39;_. The outlier types are:

*   `-1`: transition-based outlier
*   `-2`: intuitive outlier
*   `-3`: transition-based as well as intuitive outlier

With

```python
clusters, outlier_result = detector.get_outliers(tau=0.5)
```

the outliers are calculated immediately.