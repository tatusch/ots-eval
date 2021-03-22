## General Usage

In order to use the clustering algorithm C(OTS)^2 on your data, first the object has to be initialized:

```python
import pandas
from ots_eval.clustering.cots import COTS

data = pandas.DataFrame(data, columns=['object_id', 'time', 'feature1', 'feature2'])
cots = COTS(data, min_cf=0.2, sw=3)
```

Explanation of the parameter:

<figure class="table"><table><thead><tr><th>Parameter</th><th><br data-cke-filler="true"></th><th>Default</th><th>Datatype</th><th>Description</th></tr></thead><tbody><tr><td><code>data</code></td><td>-</td><td>-</td><td><i>pandas.DataFrame</i></td><td>with first column being the objectID, second being the timestamp and following columns being the features</td></tr><tr><td><code>min_cf</code></td><td>optional</td><td><i>0.015</i></td><td><i>float</i></td><td>threshold for the minimum connection factor for inserting edges to the graph</td></tr><tr><td><code>sw</code></td><td>optional</td><td><i>3</i></td><td><i>int</i></td><td>width of sliding window</td></tr></tbody></table></figure>

The names of the columns in the DataFrames are not relevant but the order of them.

The clusters can then be calculated by calling

```python
clusters = cots.create_clusters()
```

`create_clusters` returns a pandas.DataFrame with columns 'ObjectID', 'Time', features.., 'cluster' containing the data and cluster belonging/noise of all objects at all timestamps.

If clusters without noise are desired

```python
clusters = cots.get_clusters_df(min_cf=0.1, sw=5)
```

should be used. Here, `min_cf` and `sw` can be adapted if necessary.

When working with arrays, then the following methods should be used:

```python
clusters = cots.get_clusters(min_cf=0.1, sw=5)
noisy_clusters = cots.get_clusters_with_noise(min_cf=0.1, sw=5)
```

With the methods `get_factors` and `get_factors_df` the calculated factors in form of a 2-dim array or pandas.DataFrame can be investigated. Possible factor names are `'similarity', 'adaptability', 'connection', 'temporal_connection' or
'temporal_connection_sw'`, whereby `'temporal_connection_sw'` additionally needs the size of the sliding window.
                                
```python
factors = cots.get_factors(factor_type='similarity')
factors_df = cots.get_factors_df(factor_type='temporal_connection_sw', sw=3)
```                      