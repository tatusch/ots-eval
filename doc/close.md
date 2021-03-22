## General Usage

When using CLOSE as stability measure for over-time clusterings, many settings are possible. First, CLOSE has to be initialized (test\_data is a 2-dim array containing the data):

```python
import pandas
from ots_eval.stability_evaluation.close import CLOSE

data = pandas.DataFrame(test_data, columns=['object_id', 'time', 'cluster_id', 'feature1', 'feature2'])
rater = CLOSE(data, measure='mae', minPts=2, output=True, jaccard=True, weighting=True, exploitation_term=True)
```

Explanation of the parameters:

<figure class="table"><table><thead><tr><th>Parameter</th><th><br data-cke-filler="true"></th><th>Default</th><th>Datatype</th><th>Description</th></tr></thead><tbody><tr><td><code>data</code></td><td>-</td><td>-</td><td><i>pandas.DataFrame</i></td><td>with first column being the objectID, second being the timestamp, third being the clusterID and following columns being the features</td></tr><tr><td><code>measure</code></td><td>optional</td><td><i>'mae'</i></td><td><i>string</i><br><i>callable</i></td><td>describing the quality measure that should be used<br>a cluster measuring function</td></tr><tr><td><code>minPts</code></td><td>optional</td><td><i>2</i></td><td><i>int</i></td><td>used for densitiy based quality measure only</td></tr><tr><td><code>output</code></td><td>optional</td><td><i>False</i></td><td><i>boolean</i></td><td>indicating if intermediate results should be printed</td></tr><tr><td><code>jaccard</code></td><td>optional</td><td><i>False</i></td><td><i>boolean</i></td><td>indicating if jaccard index should be used in CLOSE</td></tr><tr><td><code>weighting</code></td><td>optional</td><td><i>False</i></td><td><i>boolean</i></td><td>indicating if more distant past should be weighted lower than nearer past</td></tr><tr><td><code>exploitation_term</code></td><td>optional</td><td><i>False</i></td><td><i>boolean</i></td><td>indicating if exploitation term for penalization of outliers should be used</td></tr></tbody></table></figure>

The names of the columns in the DataFrame are not relevant but the order.

Now, the clustered data set can be evaluated with CLOSE. There are two variants of quality measure:

1.  quality measure for clusters
2.  quality measure for clustering

When using the first type of quality measures, the original formula of CLOSE can be used calling the function

```python
clustering_score = rater.rate_clustering(start_time=None, end_time=None, return_measures=False)
```

where _start\_time_ and _end\_time_ indicate the time intervall which should be considered. If _start\_time_ and _end\_time_ are _None_, the first and last timestamp are considered as boundary, respectively. _return\_measures_ indicates, if the individual components of the CLOSE formula should be returned.

The second type of quality measures can be used by using the modified formula of CLOSE calling

```python
clustering_score = rater.close_t_clusterings(start_time=None, end_time=None)
```

where _start\_time_ and _end\_time_ indicate the time intervall which should be considered. If they are None, the first and last timestamp are considered as boundary, respectively.

## **Exploitation Term**

The exploitation term is originally introduced in order to penalize outliers in CLOSE.
It appends `N_co / N_o` to the CLOSE formula, where `N_co` defines the number of clustered objects and `N_o` represents the number of all objects.
When considering it as penalization term in CLOSE, it is calculated globally for the whole over-time clustering. 
But it can also be used as quality measure for example when the clusters are calculated by DBSCAN. In that case, it is computed per timestamp in order to evaluate the individual time clusterings.

### **How to use it?**

You can use the exploitation term as a penalization term by setting `exploitation_term=True` when creating the CLOSE object

```python
CLOSE(data, exploitation_term=True)
```

It is also possible to use the exploitation term as quality measure.
Therefore you have to call CLOSE as follows:

```python
CLOSE(data, measure="exploit")
```

Since the exploitation term has then to be calculated per timestamp the modified formula of CLOSE for quality measures regarding the time clusterings has to be used. Therefore, instead of using the common function `rate_clustering()` you have to call

```text
close_t_clusterings()
```

##

### Examples

#### CLOSE with DBSCAN with the exploitation term as quality measure:

```python
rater = CLOSE(data, 'exploit')
clustering_score = rater.close_t_clusterings()
```

####

#### CLOSE with DBSCAN, mean average error as quality measure and global exploitation term for outlier penalization:

```python
rater = CLOSE(data, 'mae', exploitation_term=True)
clustering_score = rater.rate_clustering()
```