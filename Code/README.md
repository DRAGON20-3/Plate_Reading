This repository contains of implementations for both Inference and Fine-Tune a model using the proposed pipeline in order to check and improve the performace, repectively. All of the codes are using the *TrOCR-x-printed model* series. 
The `Fine-Tuning` code will load the model (by downloading it from HuggingFace if it doesn't exist in the local address) first. Afterward, it will prepare the data according to parameters setting. The model will be fine-tuned using data by *Seq2SeqTrainer* class of `transformers` package from *HuggingFace*. The CER and WER will be provided for each epoch of fine-tuning and the history of them will be stored in the state file in the specific directory of the fine-tuning progress. The storing address will be printed before the fine-tuning is started.

The `Inference` code will load the model similar to Fine-Tuning and process the data based on parameters setting. Then it will use the pre-trained weights to predict the output. The predictsion will undergo the evaluation metrics consist of those mentioned in Fine-Tuning and loss. After all samples of the data exposed to the pre-trained model, the storing address for results will be printed and evaluation results will be computed and stored there.

The `Plot_History` will save a plot for the determined metric based on the state file for all of the fine-tuned models.

And, The `Sample_Evaluation` is computing the CER for each sample of validation set of each epoch using the stored files of predictions and labels in each directory of a fine-tuned model. This evaluation will be done on letters, digits and entire label (contains both letters and digits) seperately in order to provide a better evaluation of the performance of a model. The CER metrics with white spaces and without white spaces between each character will be evaluated and saved in *Sample_Evaluation_...* and *Sample_Evaluation_Processed_...* files respectively. 

The parameters in all of the codes consist of:

1. __Base_Address__: $${\color{lightblue}str}$$; The root address which contains of codes, models, and results. It SHOULD be a path instance approved with os.path.exists().
2. __Data_Address__: $${\color{lightblue}str}$$; The path to the parent folder of data which has Real Data, Synthetic Data, and Augmentation Data (for additional evaluation) directories. It SHOULD be a path instance approved with os.path.exists().
3. __Model_Name__: $${\color{lightblue}str}$$; The full name of the model from huggingface url. It supports microsoft/trocr-small-printed, microsoft/trocr-base-printed, microsoft/trocr-large-printed.
4. __Using_Synthetic_Dataset__: $${\color{lightblue}int/boolean}$$; A flag to activate using the synthetic data in the dataset of the code.
5. __Using_Augmented_Dataset__: $${\color{lightblue}int/boolean}$$; A flag enabling the augmented data to be added to the dataset of the code.
6. __Using_Real_World_Dataset__: $${\color{lightblue}int/boolean}$$; A flag to prepare real data to be utilized in the code.
7. __Validation_Size__: $${\color{lightblue}float}$$; Define the percentage of validation data from total data. The valid values are from (0.0, 1.0). For instace, 0.23 means 23% of the data is split as validation and the rest of them will be considered as training data.
8. __Dataset_Portion__: $${\color{lightblue}float}$$; This parameters controls the amount of data in the training set. The default value is 1.0. Values lower than 1.0 representing the percentage of training data to be specified as the training data. It should be in range (0.0 and 1.0]. For example, 0.75 reflect that the 75% of the training set should be used as the training set in the code. It provides more flexibility for weak equipments or debugging.
9. __Validation_Cumulation__: $${\color{lightblue}int}$$; This value reflect the method for cumulating validation data. It includes *0*: Just using the Validation Set of Real Data, *1*: Concatenating Validation et from both Real data and mentioned Synthetic data, and *2*: Just Validation set of the mentioned Synthetic data. Note that in mode 1, and 2, the Using_Synthetic_Dataset should be activated.
10. __Using_Gray_Scale_Filters__: $${\color{lightblue}int}$$; It is defining the gray scale filters that each sample of data should undergo. Moreover, it has 5 modes which consists of *0*: No filter (use RGB), *1*: Combination of filters from mode 2, 3, and 4 as a three channels image, *2*: GrayScale Filter, *3*: fastNlMeansDenoising Thresholding Filter, *4*: Sobel Edge Thresholding Filter. In mode 2, 3, and 4, the output of each filters concatenated with itself three time to form a three channel image instead of RGB input.
11. __Using_WhiteSpace__: $${\color{lightblue}int/boolean}$$; Activating the white spaces between each character of labels. It wanted the model to generate a white space between each recognized character.
12. __Style_Shifting__: $${\color{lightblue}str}$$; Determine the data transformation for synthetic data to bear resemblance to real data. This parameter has multiple values as follows that each one corresponds to one step of the proposed transformations (It is also named Style Shifting in some lines of the codes):
    - "0": No transformation .
    - "1": All of the transformations in order.
    - "2": Edge Color Padding (AKA adding colorful background).
    - "3": Adjusting Brightness and contrast randomly in a specified range Plus Blurring.
    - "4": Perspective and Stretching.
    - "5": Rotation.
    - "6": Noise Addition.
      
    It is possible to use a combination of those values; For example, "46" means using both Perspective and Stretching Plus Noise Addition. It is important to consider that the "0" or "1" are dominant values which can cover other values (e.g. all the steps will be used for either "13" or "1" because of including "1"). This implementation will assist in ablation study. Samples and some details are provided in the manuscript.

14. __Epochs__: $${\color{lightblue}int}$$; The number of training iteration for all the data
15. __Batch_Size__: $${\color{lightblue}int}$$; The number of data samples in each batch

.
