This repository contains of implementations for both Inference and Fine-Tune a model with proposed pipeline in order to check and improve the performace, repectively. All of the codes are using the TrOCR-x-printed model series. 
The Fine-Tuning code will load the model (by downloading it from HuggingFace if it doesn't exist in the local address) first. Afterward, it will prepare the data according to parameters setting. The model will be fine-tuned using data by Seq2SeqTrainer class of transformers package from HuggingFace. The CER and WER will be provided for each epoch of fine-tuning and the history of them will be stored in the state file in the specific directory of the fine-tuning progress. The storing address will be printed before the fine-tuning is started.

The Inference code will load the model similar to Fine-Tuning and process the data based on parameters setting. Then it will use the pre-trained weights to predict the output. The predictsion will undergo the evaluation metrics consist of those mentioned in Fine-Tuning and loss. After all samples of the data exposed to the pre-trained model, the storing address for results will be printed and evaluation results will be computed and stored there.

The Plot_History will save a plot for the determinded metric based on the state file for all of the fine-tuned models.

And, The Sample_Evaluation is computing the CER for each sample of validation set of each epoch using the stored files of predictions and labels in each directory of a fine-tuned model. This evaluation will be done on letters, digits and entire label (contains both letters and digits) seperately in order to provide a better evaluation of the performance of a model. The CER metrics with white spaces and without white spaces between each character will be evaluated and saved in Sample_Evaluation_... and Sample_Evaluation_Processed_... files respectively. 

The parameters in all of the codes consist of:

1.
