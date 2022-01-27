### Master Thesis Implementation

#### Description

Python implementation of the Master Thesis project "An explainable data-informed feature selection process enabling users to explore a given search space." from Leon Staubach.

Essentially, this repository implements a process to

- create several plots for data visualisation & exploration
- propose a feature ranking based on information theory
- iteratively remove features until a stop-criteria is met
- create plots to visualise the distribution of the data
- calculate metrics to assess the quality of the proposed clustering

The core-approach is an extended version of the k-Prototypes algorithm, which was firstly proposed by [Huang97] and is intended to calculate clusters from a mixed dataset of numerical and categorical variables in an unsupervised environment.
The base implementation was taken from [this](https://github.com/nicodv/kmodes#huang97) repository, but extended to handle further data-types, like compositions and cyclic datatypes.

#### Installation

Firstly, the latest development version should be cloned through

```
git clone URL
cd FOLDER
pip3 install -r requirements.txt
```

Alternatively, a virtual environment can be created to install the required dependencies.

Python 3.2 or higher is required for this repository to run properly.

After installing all the dependencies, the [configuration file](http:://PATH TO CONFIG) should be edited to reflect the local data properly. 

Right now the process requires the data-chunk to be a parquet file. Python provides simple interfaces to compress data into a parquet file. Further reading can be done on the [official documentation](https://arrow.apache.org/docs/python/parquet.html). 

Afterwards arguments should be set to determine the correct paths for inputing the data and writting the to the desired folder.