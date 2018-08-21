# clcnn-classifier
Examples of how convolutional neural networks can work with data from Aleph.

To start playing clone the folder, install Miniconda, import and activate included environment. Small raw dataset with names of Kyrgyz companies and people is included in the [data](data) folder. The same datset mixed, encoded and split is in [prepared_data](prepared_data) folder. 

Pretrained [models](models) are:

* company_person_99.h5 - distinguishes between most European and former Soviet Union personal and comapny names.
* company_person_kg.h5 - trained on the dataset provided distinguishes well between Kyrgyz personal and company names.
* male_female_96.h5 - distinguishes between most European and former Soviet Union male and female full names.

Model architecture and training example can be found at [notebooks/clcnn_classifier_model.ipynb](notebooks/clcnn_classifier_model.ipynb)
Examples of trained model classifying single and multiple unlabeled inputs are at [notebooks/predict.ipynb](notebooks/predict.ipynb)
Example of data preparation and encoding are at [notebooks/mixed_labeled_list_preparation.ipynb](notebooks/mixed_labeled_list_preparation.ipynb) and [notebooks/data_tokenization.ipynb](notebooks/data_tokenization.ipynb)

## Installation
1. [Install Miniconda](https://conda.io/miniconda.html)
1. Go to [environments](environments) folder
1. In the terminla enter: **conda env create -f clcnn.yml** (you can try **clcnn_gpu.yml** to run nneural network on your GPU, but it will work only if you have installed drivers for compatible NVIDIA GPU)
1. Activate environment with **source activate clcnn** command (for Windows: **activate clcnn**)
1. Go to [notebooks](notebooks) folder and run **jupyter notebook** command, it will open new browser window
1. Open notebook of your choice
