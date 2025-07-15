The results of comparsion between the proposed method and other similar methods are reported here. Each folder named after the methods that compared with the proposed method. 

The methods evaluated on each subset of datasets, which is described in the manuscript; so there is three files for each subset of datasets include labels, predictions, and evaluation metrics. The size of each subset is mentioned in the files name.

The evaluation metrics file contains of:
CER: Character Error Rate while characters separated with white spaces
WER: Word Error Rate while characters separated with white spaces (To check the performance on each text)
CER_Not_Split: Character Error Rate without white spaces between characters (This is reported on the manuscript as the evaluation metric)
WER_Not_Split: Word Error Rate without white spaces between characters
Google_BLEU': Calculating BLEU score proposed by Google on each sequence of characters seperated by white spaces
Google_BLEU_Not_Split': Calculating BLEU score proposed by Google on each sequence of characters with no white spaces separation
Accuracy: Computer the accuracy; for each sample, it will be 1.0 if the sequence of characters matches to the label completely.
