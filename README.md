# Over-Time Stability Evaluation
**ots-eval** is a toolset for the over-time stability evaluation of multiple multivariate time series based on cluster transitions. It contains an over-time stability measure for crisp over-time clusterings called _CLOSE_ [[1]](#1), one stability measure for fuzzy over-time clusterings called _FCSETS_ [[2]](#2), two outlier detection algorithms _DOOTS_ [[3](#3),[4](#4)] and _DACT_ [[5]](#5) addressing cluster-transition-based outliers and an over-time clustering algorithm named _C(OTS)^2_ [[6]](#6).
All approaches focus on multivariate time series data that is clustered per timestamp.

The toolset was implemented by [Martha Krakowski (Tatusch)](https://dbs.cs.uni-duesseldorf.de/mitarbeiter.php?id=tatusch) and [Gerhard Klassen](https://dbs.cs.uni-duesseldorf.de/mitarbeiter.php?id=klassen).
## Installation
You can simply install ots-eval by using pip:
```shell script
pip install ots-eval
```
You can import the package in your Python script via:
```python
import ots_eval
```
### Dependencies
ots-eval requires:
* python>=3.7
* pandas>=1.0.0
* numpy>=1.19.2
* scipy>=1.3.0

## Documentation
In the `doc` folder, there are some explanations for the usage of every approach. 
* [CLOSE](https://github.com/tatusch/ots-eval/blob/main/doc/close.md)
* [FCSETS](https://github.com/tatusch/ots-eval/blob/main/doc/fcsets.md)
* [DOOTS](https://github.com/tatusch/ots-eval/blob/main/doc/doots.md)
* [DACT](https://github.com/tatusch/ots-eval/blob/main/doc/dact.md)
* [C(OTS)^2](https://github.com/tatusch/ots-eval/blob/main/doc/cots.md)

## License
ots-eval is distributed under the 3-Clause BSD license.

## References
This toolset is the implementation of approaches from our following works:

<a id="1">[[1]](http://www.ibai-publishing.org/html/proceedings_2020/pdf/proceedings_book_MLDM_2020.pdf)</a>
Tatusch, M., Klassen, G., Bravidor, M., and Conrad, S. (2020).  
**How is Your Team Spirit? Cluster Over-Time Stability Evaluation**.  
In: _Machine Learning and Data Mining in Pattern Recognition, 16th International Conference on Machine Learning and
Data Mining, MLDM 2020_, pages 155–170.

<a id="2">[[2]](https://link.springer.com/chapter/10.1007%2F978-3-030-50146-4_50)</a>
Klassen, G., Tatusch, M., Himmelspach, L., and Conrad, S. (2020).  
**Fuzzy Clustering Stability Evaluation of Time Series**.  
In: _Information Processing and Management of Uncertainty in Knowledge-Based Systems, 18th International Conference, IPMU 2020_, pages 680-692.

<a id="3">[[3]](https://link.springer.com/chapter/10.1007/978-981-15-1699-3_8)</a>
Tatusch, M., Klassen, G., Bravidor, M., and Conrad, S. (2019).  
**Show me your friends and i’ll tell you who you are. Finding anomalous time series by conspicuous clus-
ter transitions**.  
In: _Data Mining. AusDM 2019. Communications in Computer and Information Science_, pages 91–103.

<a id="4">[[4]](https://link.springer.com/chapter/10.1007/978-3-030-59065-9_26)</a>
Tatusch, M., Klassen, G., and Conrad, S. (2020).  
**Behave or be detected! Identifying outlier sequences by their group cohesion**.  
In: _Big Data Analytics and KnowledgeDiscovery, 22nd International Conference, DaWaK 2020_, pages 333–347.

<a id="5">[[5]](https://link.springer.com/chapter/10.1007%2F978-3-030-65390-3_28)</a>
Tatusch, M., Klassen, G., and Conrad, S. (2020).  
**Loners stand out. Identification of anomalous subsequences based on group performance**.  
In: _Advanced Data Mining and Applications, ADMA 2020_, pages 360–369.

<a id="6">[[6]](https://ieeexplore.ieee.org/document/9308516) 
Klassen, G., Tatusch, M., and Conrad, S. (2020).  
**Clustering of time series regarding their over-time stability**.  
In: _Proceedings of the 2020 IEEE Symposium Series on Computational Intelligence (SSCI)_, pages 1051–1058.
