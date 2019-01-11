[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TSFDlib/TSFEL/blob/master/HAR_Example.ipynb)

# Feature Extraction Library and Classification
## Purpose
This package contains feature extraction tools and machine learning algorithms for time series data classification.

## Dependencies
Download and install the following required applications:

- [Anaconda](https://store.continuum.io/cshop/anaconda/);
- [novainstrumentation](https://github.com/hgamboa/novainstrumentation);
- [pandletstools](https://bitbucket.fraunhofer.pt/projects/SAFESENSOR/repos/pandletstools/browse).

## Feature Extraction Library
The features to extract can be defined in file */data/features.json*. The features to extract should have the parameter *"use": "yes"*.

To extract features, run:

    feature_extraction(dataset_dir, activities, device, windows_time, barometer=False)

Where **dataset_dir** is the main directory of the files (string), **activities** is the activities to discriminate (string list), **device** is the type of device (Pandlet or Smartphone) (string) and **windows_time** is the windows time in seconds (int). If you also want to consider barometer files please set **barometer=True**.

Example:

    DATASET_DIR = r'/net/sharedfolders/research_and_development/Projects/Active/DEMSports/Acquisitions/Dataset/'
    ACTIVITIES = ['Walk', 'Run', 'Stand', 'Sit']
    feature_extraction(DATASET_DIR, ACTIVITIES, 'Pandlet', 5, barometer=True)
