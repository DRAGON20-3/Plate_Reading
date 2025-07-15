import numpy as np
# import tensorflow as tf
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import time
import shutil

from IPython.display import clear_output

os.system("pip install gdown --upgrade") # !pip install gdown --upgrade
clear_output()
import gdown

import json

try:
  import accelerate
except:
  os.system("pip install accelerate") # !pip install accelerate
  clear_output()
  import accelerate



import torch
from torch.utils.data import Dataset #, DataLoader
# import torch.optim as optim
# import torchvision.transforms as transforms

# from tqdm.notebook import tqdm
# from natsort import natsorted
# import string
# import json
# import re


try:
  # from evaluate import load as load_metric
  import evaluate
except:
  os.system("pip install evaluate") # !pip install evaluate
  # !pip install datasets
  clear_output()
  # from evaluate import load as load_metric
  import evaluate

# try:
#   from bidi.algorithm import get_display
# except:
#   !pip install python-bidi
#   clear_output()
#   from bidi.algorithm import get_display

# try:
#   from arabic_reshaper import ArabicReshaper
# except:
#   !pip install arabic_reshaper
#   clear_output()
#   from arabic_reshaper import ArabicReshaper


try:
    import jiwer
    del jiwer
except:
    os.system("pip install jiwer") # !pip install jiwer
    clear_output()

try:
  from google.colab import drive
  drive.mount("/content/drive")
except:
  pass

clear_output()

####################### Global Parameters ####################### #######################
#os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in range(torch.cuda.device_count())])

#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["TORCH_USE_CUDA_DSA"] = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

Execution_Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Execution_Device = torch.device("cuda:1")
#Execution_Device = "auto"
#Execution_Device = {f"cuda:{i}": i for i in range(torch.cuda.device_count())}


Category_IDs_To_Labels = {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9"
                          , "10": "\u0623", "11": "\u0628", "12": "\u062d", "13": "\u062f", "14": "\u0631", "15": "\u0633"
                          , "16": "\u0637", "17": "\u0643", "18": "\u0644", "19": "\u0645", "20": "\u0648", "21": "\u064a"
                          , "22": "A", "23": "B", "24": "D", "25": "H", "26": "K", "27": "L", "28": "M", "29": "R", "30": "S"
                          , "31": "T", "32": "W", "33": "Y", "34": "\u0639\u064f\u0640\u0645\u0640\u0627\u0646"}
Category_Labels_To_IDs = {i[1]:i[0] for i in Category_IDs_To_Labels.items()}

Base_Address = "/content/drive/MyDrive/Neural Network/Plate_Reading/data/"
# Base_Address = "/content/"
Base_Address = "./Data/"
Data_Address = "/content/Data/"
# Data_Address = "/content/drive/MyDrive/Neural Network/Plate_Reading/data/"
Data_Address = "./Data/"

Model_Name = "microsoft/trocr-small-printed"
#Model_Name = "microsoft/trocr-base-printed"
Model_Name = "microsoft/trocr-large-printed"

Using_Synthetic_Dataset = 0
Using_White_BackGround_Synthetic_Dataset = 0
Using_Black_BackGround_Synthetic_Dataset = 0
Using_Augmented_Dataset = 1
Using_Real_World_Dataset = 0 # Dubizle (Or Open Souq), Labels Are Without Arabic Letters


Dataset_Portion = 1.0 # 0.25 # ??? ??? For Each One Separately Or After Combination

Validation_Size = 0.2 # 0.3 # 0.23 means 23%
Validation_Cumulation = 0 # 0: Just Using the Fixed Evaludation Set Of Real Data, 1: Concatenate The Fixed Evaludation Set Of Real Data with Validation Set of The Mentioned Section

Using_Gray_Scale_Filters = 0 # Using Grayscale and Filters Over The Image Instead Of RGB Image

Epochs = 10 # 10 # Training_Epochs
Batch_Size = 32

####################### History Plotting ####################### #######################


# trocr-small-printed_1739107801_Fine-Tuning
Models_Name = [name for name in os.listdir(Base_Address + "Model/") if (("trocr" in name.lower() and "fine-tun" in name.lower()) and (os.path.isfile(Base_Address + "Model/" + name + "/trainer_state.json")))]
Models_Name.sort()

# Plot_Metric = "CER" # "WER", "Loss"
Plot_Metric = "loss"

# "loss": [0.0, 23.365478515625]
# "CER": [0.010972716488730723, 2.716488730723606]
# "WER": [0.01579520697167756, 2.863289760348584]


# plt.rcParams.update({"font.family": "Times New Roman", "font.size": 23}) # WARNING:matplotlib.font_manager:findfont: Font family 'Times New Roman' not found.
plt.rcParams.update({"font.family": "serif", "font.serif": ['Times New Roman'] + plt.rcParams['font.serif'], "font.size": 23})
#

for j in range(len(Models_Name)):
  #
  F = open(Base_Address + "Model/" + Models_Name[j] + "/trainer_state.json", "r")
  History = json.loads(F.read())
  # History = F.read().replace(" ", "").split("\n")
  F.close()

  if sum([int(Plot_Metric in i.keys()) for i in History["log_history"]]) == 0 and sum([int("eval_" + Plot_Metric in i.keys()) for i in History["log_history"]]) == 0:
    print("Metric Is Not Found!!")
    continue
  #
  plt.figure(figsize = (20, 8))
  # plt.title()
  try:
    Temp = [[History["log_history"][i]["epoch"], History["log_history"][i][Plot_Metric]] for i in range(len(History["log_history"])) if Plot_Metric in History["log_history"][i].keys()]
    Temp = np.array(Temp)
    # plt.plot(Temp[:, 0], Temp[:, 1], linestyle = "-", linewidth = 3.1, label = "Training " + Plot_Metric if Plot_Metric == Plot_Metric.upper() else Plot_Metric.title())
    plt.plot(Temp[:, 0], Temp[:, 1], linestyle = "-", linewidth = 3.1, label = "Training")
  except:
    print("The Metrics Does Not Exist In The Training History!!")
  #
  try:
    Temp = [[History["log_history"][i]["epoch"], History["log_history"][i]["eval_" + Plot_Metric]] for i in range(len(History["log_history"])) if "eval_" + Plot_Metric in History["log_history"][i].keys()]
    Temp = np.array(Temp)
    # plt.plot(Temp[:, 0], Temp[:, 1], linestyle = "-", linewidth = 3.1, label = "Validation " + Plot_Metric if Plot_Metric == Plot_Metric.upper() else Plot_Metric.title())
    plt.plot(Temp[:, 0], Temp[:, 1], linestyle = "-", linewidth = 3.1, label = "Validation")
  except:
    print("The Metrics Does Not Exist In The Validation History!!")

  #
  # Plot_Metric = Plot_Metric if Plot_Metric == Plot_Metric.upper() else Plot_Metric.title()
  #
  plt.grid()
  plt.legend()
  plt.xlabel("Epochs")
  plt.ylabel(Plot_Metric if Plot_Metric == Plot_Metric.upper() else Plot_Metric.title())
  #
  plt.tight_layout()
  # plt.show()
  plt.savefig(Base_Address + "Model/" + Models_Name[j] + "/History_" + str(Plot_Metric if Plot_Metric == Plot_Metric.upper() else Plot_Metric.title()) + ".png")
  plt.close()

print()
