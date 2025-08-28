The Results of the experiments in the <a href = "">maniscript</a> is reported here.

There are three objectives in the experiments as the following:
1. __Inference__: Using pre-trained or publicly available models on the datasets to check their performance and provide a baseline.

2. __Fine-Tuning__: Reporting the parameters setting, history of metrics in training progress, and evaluation metrics. This experiments are separated according to the tables of the manuscript.

3. __Comparison__: This includes the measured performance of other methodology, reported in the manuscript, in order to be comparable with the proposed methodology.


Also, Some files exist in those directories consist of:
- Label: a text format file contains the each label of a dataset in a line. Note that the number of samples is reported in the filename which refers to the subsets of datasets described in the manuscript. Also, if the number of samples isn't reported in the filename, it should be considered as the Validation set of Real Data (Known as Subset II of Real Data in the paper).

- Prediction: a text format with the same style and naming pattern of the Label file. This file contains of predictions separated in lines. This will make you able to compare the labels and predictions line by line for each sample in a dataset.

- Evaluation_Metric: this file is also a text file which consists of measured values of various metrics as a Dictionary. The dataset of this evaluation could be found based on the number of samples in the filename (e.g. Evaluation_Metrics_500.txt refers to the Subset II or the Real data which is known as Validation set also.).

. 
