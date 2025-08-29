The Results of the experiments in the <a href = "">maniscript</a> is reported here.

There are three objectives in the experiments as the following:
1. __Inference__: Using pre-trained or publicly available models on the datasets to check their performance and provide a baseline.

2. __Fine-Tuning__: Reporting the parameters setting, history of metrics in training progress, and evaluation metrics. This experiments are separated according to the tables of the manuscript.

3. __Comparison__: This includes the measured performance of other methodology, reported in the manuscript, in order to be comparable with the proposed methodology.


Some files exist in those directories consist of:
- *Label*: a text format file contains the each label of a dataset in a line. Note that the number of samples is reported in the filename which refers to the subsets of datasets described in the manuscript. Also, if the number of samples isn't reported in the filename, it should be considered as the Validation set of Real Data (Known as Subset II of Real Data in the paper).

- *Prediction*: a text format with the same style and naming pattern of the Label file. This file contains of predictions separated in lines. This will make you able to compare the labels and predictions line by line for each sample in a dataset.

- *Evaluation_Metric*: this file is also a text file which consists of measured values of various metrics as a Dictionary. The dataset of this evaluation could be found based on the number of samples in the filename (e.g. Evaluation_Metrics_500.txt refers to the Subset II or the Real data which is known as Validation set also.).


Additionally, there are other files which could be found in various experiments according to objective of that experiment of providing better evaluation. They include:

- *History*: an image include a plot for measeurement of a metrics in each step of fine-tuning. For instance, History_Loss.png is the history of loss in the fine-tuning progress.

- *Parameters*: This textfile exists in Fine-Tuning and Inference experiment and it comprises the parameter setting of the executed code. The detailed explanation of them is provided in the `Code` repository.

- *Sample_Evaluation*: A sheet file contains sample-wise evaluation using CER (Character Error Rate) and Accuracy. There are three evaluaion for each sample in this file include All_Characters, Digits, and Letters. Regarding the evaluation of a model performance, this evaluation will show whether the model preform better on digit recognition, letter recognition or both. The size of the dataset is mentioned in the filename.

- *Sample_Evaluation_Processed*: This file is almost similar to the Sample_Evaluation files except some processing on the predictions. As the maximum number of digits and letters in a plate of this dataset are 5 and 2 respectively, another option to interpret and evaluate predictions is to choose the first 5 digits and first 2 letters and ignore the other values; Beside providing another view for evaluation, this method will prevent the CER values over 1.0 which would happen because of longer length of text in a prediction other than a label.



. 
