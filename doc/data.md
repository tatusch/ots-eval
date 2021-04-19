## Provided Data Sets

The folder `data` provides the generated data sets from our listed publications. The filenames refer to the publication numbering in the main README file.

`data/pub1_generated_datasetA.csv` refers for example to data set A from the publication *How is Your Team Spirit? Cluster Over-Time Stability Evaluation* (see main README).

The columns in the CSV files are separated by commas, so they can easily be loaded using pandas:
```python
import pandas

data = pandas.read_csv(<filename>)
```

The column names are `object_id, time, feature1` and in case of 2d data `feature2`.