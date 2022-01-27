# Master Thesis Implementation

### Description

Python implementation of the Master Thesis project "An explainable data-informed feature selection process enabling users to explore a given search space." from Leon Staubach.

Essentially, this repository implements a process to

- create several plots for data visualisation & exploration
- propose a feature ranking based on information theory
- iteratively remove features until a stop-criteria is met
- create plots to visualise the distribution of the data
- calculate metrics to assess the quality of the proposed clustering

The core-approach is an extended version of the k-Prototypes algorithm, which was firstly proposed by [Huang97] and is intended to calculate clusters from a mixed dataset of numerical and categorical variables in an unsupervised environment.
The base implementation was taken from [this](https://github.com/nicodv/kmodes#huang97) repository, but extended to handle further data-types, like compositions and cyclic datatypes.

### Installation

Firstly, the latest development version should be cloned through

```
git clone URL
cd FOLDER
pip3 install -r requirements.txt
```

Alternatively, a virtual environment can be created to install the required dependencies.

Python 3.2 or higher is required for this repository to run properly.

After installing all the dependencies, the configuration file should be edited to reflect the local data properly. An example for a config file was uploaded [here](https://github.com/leonstaubach/Explainable-Data-Discovery/blob/main/uploadable_config.py). It is highly suggested to create a own file called `local_config.py` and copy all the attributues from the example. This will automatically be used from the system.

Right now the process requires the data-chunk to be a parquet file. Python provides simple interfaces to compress data into a parquet file. Further reading can be done on the [official documentation](https://arrow.apache.org/docs/python/parquet.html). 

### Usage

After the configuration is ready and the data was moved to the correct path, the main process can be triggered by running this command in a terminal

```python
python3 main.py
```

Logs and output images will be automatically stored in the configured location.

### Important Notes

As described in the top section, this process is intended to be an extensive exploration tool. Data quality needs to be assumed, the nature of unsupervised learning does not allow a process to pursure targeted learning.
The underlying algorithm tries to minimize distances from data points to centroids (refer to k-Means), therefore flat geometry will be assumed for the data.
This process might not be optimal, if:

- Flat geometry does not describe the data well
- Outliers are common (since they are not detectable by k-Prototypes)
- No mixed-data (classic approaches are usually more established and researched on)

### Outlook

##### Code

Currently, the data is required to be in parquet format. Future efforts should be used to provide a simple interface for different data sources, including `csv` Files or common database connectors.

Furthermore a single k-Prototypes process should be executable in parallel to decrease runtime.

##### Research

Generally speaking, the field of mixed-data clustering is a field that attracts less attention from modern research. Simply distinguishing between numerical and categorical variables and transforming one into the other to have a homogenous datatype, is the most common practice. There are good reasons for that, to name a few:

- data can be aggregated easily
- calculations are vectorizable and easily parallelizable
- especially for numerical variables, distance assumptions allows sophisticated approaches to be used like embedded clustering

Keeping data in its existing form without applying transformation to change the semantics of the variable, can lead to alot less information loss and less wrong conclusions drawn from data. 

In the future some important tasks are

- Applying mixed-data logic on other existing clustering algorithms (like DBSCan)
- More sophisticated way of aggregating different datatypes for a fair comparison

