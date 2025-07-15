import numpy as np
import pandas as pd
import time
import os

import jiwer


Base_Address = "/content/drive/MyDrive/Neural Network/Plate_Reading/data/"
# Base_Address = "/content/"
Base_Address = "./Data/"


####################### Computing Evaluation Metrics For Each Sample #######################

# Models_Name = [name for name in os.listdir(Base_Address + "Model/") if ("My-Model".lower() in name.lower() and (os.path.isfile(Base_Address + "Model/" + name + "/Labels.txt") and os.path.isfile(Base_Address + "Model/" + name + "/Predictions.txt")))]
Models_Name = [name for name in os.listdir(Base_Address + "Model/") if ("trocr" in name.lower() and (os.path.isfile(Base_Address + "Model/" + name + "/Labels.txt") and os.path.isfile(Base_Address + "Model/" + name + "/Predictions.txt")))]
Models_Name.sort()

# valid_letters = {'أ': 'A', 'ب': 'B', 'د': 'D', 'ر': 'R', 'س': 'S', 'ك': 'K', 'م': 'M', 'و': 'W', 'ي': 'Y', 'ح': 'H', 'ط': 'T', 'ل': 'L'}

Columns = ["Type", "Label", "Prediction", "CER", "Accuracy"] # Type : ["All_Characters", "Digits", "Letters"]
Results = []

for j in range(len(Models_Name)):
  F = open(Base_Address + "Model/" + Models_Name[j] + "/Labels.txt", "r")
  Labels = F.read().split("\n")
  # Labels = F.read().replace(" ", "").split("\n")
  F.close()
  #
  F = open(Base_Address + "Model/" + Models_Name[j] + "/Predictions.txt", "r")
  Predictions = F.read().replace("<unk>", "").split("\n")
  # Predictions = F.read().replace("<unk>", "").replace("  ", "").split("\n") #
  # Predictions = F.read().replace("<unk>", "").replace(" ", "").split("\n")
  Predictions = [i.strip() for i in Predictions]
  F.close()
  #
  ######### Not Processed Prediction #########
  Results = []
  #
  Temp = len(Results)
  Results += [["All_Characters", Labels[i], Predictions[i], jiwer.cer(reference = Labels[i], hypothesis = Predictions[i]), int(Labels[i].lower() == Predictions[i].lower())] for i in range(len(Labels))]
  Results += [["All_Characters", "Average", "Average", np.mean(np.array(Results)[Temp:, 3].astype(np.float64)), np.mean(np.array(Results)[Temp:, 4].astype(np.float64))]]
  #
  Temp = len(Results)
  Temp_Labels = ["".join([k for k in i if k.isdigit()]) for i in Labels]
  Temp_Predictions = ["".join([k for k in i if k.isdigit()]) for i in Predictions]
  Results += [["Digits", Temp_Labels[i], Temp_Predictions[i], jiwer.cer(reference = Temp_Labels[i], hypothesis = Temp_Predictions[i]), int(Temp_Labels[i].lower() == Temp_Predictions[i].lower())] for i in range(len(Temp_Labels))]
  Results += [["Digits", "Average", "Average", np.mean(np.array(Results)[Temp:, 3].astype(np.float64)), np.mean(np.array(Results)[Temp:, 4].astype(np.float64))]]
  #
  Temp = len(Results)
  Temp_Labels = ["".join([k for k in i if (not k.isdigit() and k != " ")]) for i in Labels]
  Temp_Predictions = ["".join([k for k in i if (not k.isdigit() and k != " ")]) for i in Predictions]
  #
  Temp_Predictions = [Temp_Predictions[i] for i in range(len(Temp_Predictions)) if i not in np.argwhere(np.char.replace(np.array(Temp_Labels), " ", "") == "")[:, 0].tolist()]
  Temp_Labels = [Temp_Labels[i] for i in range(len(Temp_Labels)) if i not in np.argwhere(np.char.replace(np.array(Temp_Labels), " ", "") == "")[:, 0].tolist()]
  Results += [["Letters", Temp_Labels[i], Temp_Predictions[i], jiwer.cer(reference = Temp_Labels[i], hypothesis = Temp_Predictions[i]), int(Temp_Labels[i].lower() == Temp_Predictions[i].lower())] for i in range(len(Temp_Labels))]
  Results += [["Letters", "Average", "Average", np.mean(np.array(Results)[Temp:, 3].astype(np.float64)), np.mean(np.array(Results)[Temp:, 4].astype(np.float64))]]
  #
  #
  Results = pd.DataFrame(Results, columns = Columns)
  Results.to_csv(Base_Address + "Model/" + Models_Name[j] + "/Sample_Evaluation_" + str(len(Labels)) + ".csv", index = False)
  #
  ######### Processed Prediction #########
  Labels = [i.replace(" ", "") for i in Labels]
  Predictions = [i.replace(" ", "") for i in Predictions]
  #
  Results = []
  #
  Temp = len(Results)
  Results += [["All_Characters", Labels[i], Predictions[i], jiwer.cer(reference = Labels[i], hypothesis = Predictions[i]), int(Labels[i].lower() == Predictions[i].lower())] for i in range(len(Labels))]
  Results += [["All_Characters", "Average", "Average", np.mean(np.array(Results)[Temp:, 3].astype(np.float64)), np.mean(np.array(Results)[Temp:, 4].astype(np.float64))]]
  #
  Temp = len(Results)
  Temp_Labels = ["".join([k for k in i[:5] if k.isdigit()]) for i in Labels]
  Temp_Predictions = ["".join([k for k in i[:5] if k.isdigit()]) for i in Predictions]
  Results += [["Digits", Temp_Labels[i], Temp_Predictions[i], jiwer.cer(reference = Temp_Labels[i], hypothesis = Temp_Predictions[i]), int(Temp_Labels[i].lower() == Temp_Predictions[i].lower())] for i in range(len(Temp_Labels))]
  Results += [["Digits", "Average", "Average", np.mean(np.array(Results)[Temp:, 3].astype(np.float64)), np.mean(np.array(Results)[Temp:, 4].astype(np.float64))]]
  #
  Temp = len(Results)
  Temp_Labels = ["".join([k for k in i[:7] if (not k.isdigit() and ord(k) < 128)]) for i in Labels]
  Temp_Predictions = ["".join([k for k in i[:7] if (not k.isdigit() and ord(k) < 128)]) for i in Predictions]
  # Temp_Predictions = ["".join([k for k in i[:7] if (not k.isdigit() and ord(k) < 128)][:2]) for i in Predictions] # Force To Just 2 Letters In The Prediction
  #
  Temp_Predictions = [Temp_Predictions[i] for i in range(len(Temp_Predictions)) if i not in np.argwhere(np.array(Temp_Labels) == "")[:, 0].tolist()]
  Temp_Labels = [Temp_Labels[i] for i in range(len(Temp_Labels)) if i not in np.argwhere(np.array(Temp_Labels) == "")[:, 0].tolist()]
  Results += [["Letters", Temp_Labels[i], Temp_Predictions[i], jiwer.cer(reference = Temp_Labels[i], hypothesis = Temp_Predictions[i]), int(Temp_Labels[i].lower() == Temp_Predictions[i].lower())] for i in range(len(Temp_Labels))]
  Results += [["Letters", "Average", "Average", np.mean(np.array(Results)[Temp:, 3].astype(np.float64)), np.mean(np.array(Results)[Temp:, 4].astype(np.float64))]]
  #
  #
  Results = pd.DataFrame(Results, columns = Columns)
  Results.to_csv(Base_Address + "Model/" + Models_Name[j] + "/Sample_Evaluation_Processed_" + str(len(Labels)) + ".csv", index = False)
  print(Models_Name[j], "Is Evaluated")
  print("----------------------- ----------------------- -----------------------")

#

print()
