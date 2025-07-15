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

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

#Model_Name = "microsoft/trocr-small-printed"
#Model_Name = "microsoft/trocr-base-printed"
Model_Name = "microsoft/trocr-large-printed"

Using_Synthetic_Dataset = 0
Using_White_BackGround_Synthetic_Dataset = 0
Using_Black_BackGround_Synthetic_Dataset = 0
Using_Augmented_Dataset = 1
Using_Real_World_Dataset = 0 # Dubizle (Or Open Souq), Labels Are Without Arabic Letters


Dataset_Portion = 1.0 # 0.25 # ??? ??? For Each One Separately Or After Combination

Validation_Size = 0.2 # 0.3 # 0.23 means 23%
Validation_Cumulation = 2 # 0: Just Using the Fixed Evaludation Set Of Real Data, 1: Concatenate The Fixed Evaludation Set Of Real Data with Validation Set of The Mentioned Section

Using_Gray_Scale_Filters = 0 # Using Grayscale and Filters Over The Image Instead Of RGB Image
Using_WhiteSpace = 0 # Using WhiteSpace In Tokenization Between Each Letter (1: "1 3 4 B", 0: "134B")

Epochs = 10 # 10 # Training_Epochs
Batch_Size = 32

####################### Data Loading ####################### #######################

if Using_Synthetic_Dataset == 1 or Using_White_BackGround_Synthetic_Dataset == 1 or Using_Black_BackGround_Synthetic_Dataset == 1 or Using_Augmented_Dataset == 1:
  if not os.path.isdir(Data_Address + "image-generation-and-augmentation/"):
    try:
      shutil.unpack_archive("/content/drive/MyDrive/Neural Network/Plate_Reading/data/image-generation-and-augmentation.zip", Data_Address)
    except:
      gdown.download(id = "12VLkg9Tez6L808KhwIqAhP_kykMMwfFe") # https://drive.google.com/file/d/12VLkg9Tez6L808KhwIqAhP_kykMMwfFe/view?usp=sharing
      # gdown.download(id = "1YzrEwfskq3SfZSvKBn7IIIHEVc1_zPC6") # https://drive.google.com/file/d/1YzrEwfskq3SfZSvKBn7IIIHEVc1_zPC6/view?usp=sharing
      shutil.unpack_archive(os.getcwd() + "/image-generation-and-augmentation.zip", Data_Address)

Data_Train = pd.DataFrame([])
Data_Test = pd.DataFrame([])

if Using_Synthetic_Dataset == 1:
  Temp = pd.read_pickle(Data_Address + "image-generation-and-augmentation/evaluation/trocr/gen_images.df")
  np.random.seed(23)
  Temp_2 = np.random.choice(Temp.shape[0], size = Temp.shape[0], replace = False)
  # Data = pd.concat([Data, Temp], ignore_index = True)
  Data_Train = pd.concat([Data_Train, Temp.iloc[Temp_2[:int(-1 * (Validation_Size * Temp_2.shape[0]))]] ], ignore_index = True)
  Data_Test = pd.concat([Data_Test, Temp.iloc[Temp_2[int(-1 * (Validation_Size * Temp_2.shape[0])):]] ], ignore_index = True)

if Using_White_BackGround_Synthetic_Dataset == 1:
  Temp = pd.read_pickle(Data_Address + "image-generation-and-augmentation/evaluation/trocr/gen_no_ar_w_images.df")
  np.random.seed(23)
  Temp_2 = np.random.choice(Temp.shape[0], size = Temp.shape[0], replace = False)
  # Data = pd.concat([Data, Temp], ignore_index = True)
  Data_Train = pd.concat([Data_Train, Temp.iloc[Temp_2[:int(-1 * (Validation_Size * Temp_2.shape[0]))]] ], ignore_index = True)
  Data_Test = pd.concat([Data_Test, Temp.iloc[Temp_2[int(-1 * (Validation_Size * Temp_2.shape[0])):]] ], ignore_index = True)

if Using_Black_BackGround_Synthetic_Dataset == 1:
  Temp = pd.read_pickle(Data_Address + "image-generation-and-augmentation/evaluation/trocr/gen_no_ar_k_images.df")
  np.random.seed(23)
  Temp_2 = np.random.choice(Temp.shape[0], size = Temp.shape[0], replace = False)
  # Data = pd.concat([Data, Temp], ignore_index = True)
  Data_Train = pd.concat([Data_Train, Temp.iloc[Temp_2[:int(-1 * (Validation_Size * Temp_2.shape[0]))]] ], ignore_index = True)
  Data_Test = pd.concat([Data_Test, Temp.iloc[Temp_2[int(-1 * (Validation_Size * Temp_2.shape[0])):]] ], ignore_index = True)

if Using_Augmented_Dataset == 1:
  Temp = pd.read_pickle(Data_Address + "image-generation-and-augmentation/evaluation/trocr/aug_images.df")
  np.random.seed(23)
  Temp_2 = np.random.choice(Temp.shape[0], size = Temp.shape[0], replace = False)
  # Data = pd.concat([Data, Temp], ignore_index = True)
  Data_Train = pd.concat([Data_Train, Temp.iloc[Temp_2[:int(-1 * (Validation_Size * Temp_2.shape[0]))]] ], ignore_index = True)
  Data_Test = pd.concat([Data_Test, Temp.iloc[Temp_2[int(-1 * (Validation_Size * Temp_2.shape[0])):]] ], ignore_index = True)
  #

if Using_Real_World_Dataset == 1:
  #
  if not os.path.isdir(Data_Address + "labeling-helper-source/db/"):
    gdown.download(id = "1D9XJYpU6f5lLOGDBFNSRZswDxZ2KLTT_") # https://drive.google.com/file/d/1D9XJYpU6f5lLOGDBFNSRZswDxZ2KLTT_/view?usp=sharing
    shutil.unpack_archive(os.getcwd() + "/labeling-helper-source.zip", Data_Address)
    if os.path.isdir(Data_Address + "labeling-helper-source/Dubizzle_Data/plate_extraction_conf=0.6/"):
      os.rename(Data_Address + "labeling-helper-source/Dubizzle_Data/plate_extraction_conf=0.6/", Data_Address + "labeling-helper-source/Dubizzle_Data/plate_extraction;conf=0.6/")
  #
  if not os.path.isfile(Data_Address + "labeling-helper-source/Dubizzle_Training_Data.npz"):
    gdown.download(id = "1sYHXHOR4PC8nGkgY2FP51pVo5t8El0Hn", output = Data_Address + "labeling-helper-source/Dubizzle_Training_Data.npz") # Training Data: https://drive.google.com/file/d/1sYHXHOR4PC8nGkgY2FP51pVo5t8El0Hn/view?usp=sharing
  #
  if not os.path.isfile(Data_Address + "labeling-helper-source/Dubizzle_Evaluation_Data.npz"):
    gdown.download(id = "1gp4Hg2Vi3DH4Zx3EgnsGPqObd3j8k9CD", output = Data_Address + "labeling-helper-source/Dubizzle_Evaluation_Data.npz") # Evaluation Data: https://drive.google.com/file/d/1gp4Hg2Vi3DH4Zx3EgnsGPqObd3j8k9CD/view?usp=sharing
  #
  Temp = np.load(Data_Address + "labeling-helper-source/Dubizzle_Training_Data.npz", allow_pickle = True)["Data"] # Training Data
  Temp = pd.DataFrame(Temp, columns = ["file_path", "text", "bbox_path"])
  Evaluation_Data = np.load(Data_Address + "labeling-helper-source/Dubizzle_Evaluation_Data.npz", allow_pickle = True)["Data"] # Evaluation Data
  Evaluation_Data = pd.DataFrame(Evaluation_Data, columns = ["file_path", "text", "bbox_path"])
  #
  Temp["file_path"] = np.char.replace(Temp["file_path"].to_numpy().astype("<U230"), "/content/drive/MyDrive/Neural Network/Plate_Reading/data/labeling-helper-source/", Data_Address + "labeling-helper-source/")
  Temp["bbox_path"] = np.char.replace(Temp["bbox_path"].to_numpy().astype("<U230"), "/content/drive/MyDrive/Neural Network/Plate_Reading/data/labeling-helper-source/", Data_Address + "labeling-helper-source/")
  #
  Evaluation_Data["file_path"] = np.char.replace(Evaluation_Data["file_path"].to_numpy().astype("<U230"), "/content/drive/MyDrive/Neural Network/Plate_Reading/data/labeling-helper-source/", Data_Address + "labeling-helper-source/")
  Evaluation_Data["bbox_path"] = np.char.replace(Evaluation_Data["bbox_path"].to_numpy().astype("<U230"), "/content/drive/MyDrive/Neural Network/Plate_Reading/data/labeling-helper-source/", Data_Address + "labeling-helper-source/")
  #
  Temp_2 = None
  #
  Data_Train = pd.concat([Data_Train, Temp], ignore_index = True)
  Data_Test = pd.concat([Data_Test, Evaluation_Data], ignore_index = True)
  #

del Temp, Temp_2

if "Evaluation_Data" not in globals():
  if not os.path.isfile(Data_Address + "labeling-helper-source/Dubizzle_Evaluation_Data.npz"):
    gdown.download(id = "1gp4Hg2Vi3DH4Zx3EgnsGPqObd3j8k9CD", output = Data_Address + "labeling-helper-source/Dubizzle_Evaluation_Data.npz") # Evaluation Data: https://drive.google.com/file/d/1gp4Hg2Vi3DH4Zx3EgnsGPqObd3j8k9CD/view?usp=sharing
  #
  Evaluation_Data = np.load(Data_Address + "labeling-helper-source/Dubizzle_Evaluation_Data.npz", allow_pickle = True)["Data"] # Evaluation Data
  Evaluation_Data = pd.DataFrame(Evaluation_Data, columns = ["file_path", "text", "bbox_path"])
  #
  Evaluation_Data["file_path"] = np.char.replace(Evaluation_Data["file_path"].to_numpy().astype("<U230"), "/content/drive/MyDrive/Neural Network/Plate_Reading/data/labeling-helper-source/", Data_Address + "labeling-helper-source/")
  Evaluation_Data["bbox_path"] = np.char.replace(Evaluation_Data["bbox_path"].to_numpy().astype("<U230"), "/content/drive/MyDrive/Neural Network/Plate_Reading/data/labeling-helper-source/", Data_Address + "labeling-helper-source/")

if "Dataset_Portion" in globals() and Dataset_Portion < 1.0:
  np.random.seed(23)
  Temp = np.random.choice(Data_Train.shape[0], size = int(Dataset_Portion * Data_Train.shape[0]), replace = False)
  Data_Train = Data_Train.iloc[Temp]
  del Temp

if Data_Address not in Data_Train["file_path"].iloc[-1]:
  # for i in range(Data_Train.shape[0]):
  #   Data_Train["file_path"].iloc[i] = Data_Train["file_path"].iloc[i].replace("/home/as53480/om/lp/", Data_Address)
  #   Data_Train["bbox_path"].iloc[i] = Data_Train["bbox_path"].iloc[i].replace("/home/as53480/om/lp/", Data_Address)
  Data_Train["file_path"] = np.char.replace(Data_Train["file_path"].to_numpy().astype("<U230"), "/home/as53480/om/lp/", Data_Address)
  Data_Train["bbox_path"] = np.char.replace(Data_Train["bbox_path"].to_numpy().astype("<U230"), "/home/as53480/om/lp/", Data_Address)

if Data_Address not in Data_Test["file_path"].iloc[-1]:
  # for i in range(Data_Test.shape[0]):
  #   Data_Test["file_path"].iloc[i] = Data_Test["file_path"].iloc[i].replace("/home/as53480/om/lp/", Data_Address)
  #   Data_Test["bbox_path"].iloc[i] = Data_Test["bbox_path"].iloc[i].replace("/home/as53480/om/lp/", Data_Address)
  Data_Test["file_path"] = np.char.replace(Data_Test["file_path"].to_numpy().astype("<U230"), "/home/as53480/om/lp/", Data_Address)
  Data_Test["bbox_path"] = np.char.replace(Data_Test["bbox_path"].to_numpy().astype("<U230"), "/home/as53480/om/lp/", Data_Address)


if "/content/Data/" in Data_Train["file_path"].iloc[-1] or "/content/Data/" in Evaluation_Data["file_path"].iloc[-1]:
	#
	Data_Train["file_path"] = np.char.replace(Data_Train["file_path"].to_numpy().astype("<U230"), "/content/Data/", Data_Address)
	Data_Test["file_path"] = np.char.replace(Data_Test["file_path"].to_numpy().astype("<U230"), "/content/Data/", Data_Address)
	Evaluation_Data["file_path"] = np.char.replace(Evaluation_Data["file_path"].to_numpy().astype("<U230"), "/content/Data/", Data_Address)
	Data_Train["bbox_path"] = np.char.replace(Data_Train["bbox_path"].to_numpy().astype("<U230"), "/content/Data/", Data_Address)
	Data_Test["bbox_path"] = np.char.replace(Data_Test["bbox_path"].to_numpy().astype("<U230"), "/content/Data/", Data_Address)
	Evaluation_Data["bbox_path"] = np.char.replace(Evaluation_Data["bbox_path"].to_numpy().astype("<U230"), "/content/Data/", Data_Address)


######### Remove Non-English Letters #########
Data_Train.iloc[:, 1] = np.char.decode(np.char.encode(Data_Train.iloc[:, 1].to_numpy().astype(np.str_), encoding = "ascii", errors = "ignore"))
Data_Train.iloc[:, 1] = np.char.replace(np.char.strip(Data_Train.iloc[:, 1].to_numpy().astype(np.str_)), "  ", " ")

Data_Test.iloc[:, 1] = np.char.decode(np.char.encode(Data_Test.iloc[:, 1].to_numpy().astype(np.str_), encoding = "ascii", errors = "ignore"))
Data_Test.iloc[:, 1] = np.char.replace(np.char.strip(Data_Test.iloc[:, 1].to_numpy().astype(np.str_)), "  ", " ")

Evaluation_Data.iloc[:, 1] = np.char.decode(np.char.encode(Evaluation_Data.iloc[:, 1].to_numpy().astype(np.str_), encoding = "ascii", errors = "ignore"))
Evaluation_Data.iloc[:, 1] = np.char.replace(np.char.strip(Evaluation_Data.iloc[:, 1].to_numpy().astype(np.str_)), "  ", " ")
######### #########

print("Data_Train Shape:", Data_Train.shape)
print("Data_Test Shape:", Data_Test.shape)
print("Evaluation_Data Shape:", Evaluation_Data.shape)

"""
if Data_Address not in Data["bbox_path"].iloc[-1]:
  for i in range(Data.shape[0]):
    Data["file_path"].iloc[i] = Data["file_path"].iloc[i].replace("/home/as53480/om/lp/", Data_Address)
    Data["bbox_path"].iloc[i] = Data["bbox_path"].iloc[i].replace("/home/as53480/om/lp/", Data_Address)


np.random.seed(23)
# Shuffled_Indices = np.random.choice(np.arange(Data.shape[0]), Data.shape[0], replace = False)
# Shuffled_Indices = np.random.choice(np.arange(Data.shape[0]), int(Data.shape[0] / 4), replace = False)
Shuffled_Indices = np.random.choice(np.arange(Data.shape[0]), int(Dataset_Portion * Data.shape[0]), replace = False)


# Train_Data = Data.iloc[Shuffled_Indices[: int((1 - Validation_Size) * Shuffled_Indices.shape[0])], :]
# Validation_Data = Data.iloc[Shuffled_Indices[int((1 - Validation_Size) * Shuffled_Indices.shape[0]):], :]
Train_Data = Data.iloc[Shuffled_Indices[: -1 * int(Validation_Size * Shuffled_Indices.shape[0])], :]
Validation_Data = Data.iloc[Shuffled_Indices[-1 * int(Validation_Size * Shuffled_Indices.shape[0]):], :]
"""

Train_Data = Data_Train.copy()
Validation_Cumulation = 2 # 2: Inferencing

if Validation_Cumulation == 0:
  Validation_Data = Evaluation_Data.copy()
elif Validation_Cumulation == 1:
  Validation_Data = pd.concat([Evaluation_Data, Data_Test], ignore_index = True)
elif Validation_Cumulation == 2: # Inferencing
  Validation_Data = Data_Test.copy()

del Data_Train, Data_Test

print()
####################### Loading TrOCR Model ####################### #######################
##torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

from transformers import VisionEncoderDecoderModel, TrOCRProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator #, pipeline

Processor = TrOCRProcessor.from_pretrained(Model_Name) # , device_map = Execution_Device

##Execution_Device = {"encoder": torch.device("cuda:1"), "decoder": torch.device("cuda:1")}

try:
  print(Skip_Parameter)
  Model = VisionEncoderDecoderModel.from_pretrained(Base_Address + "Transformers/" + Model_Name.replace("/" , "_") + "/") # , device_map = 	Execution_Device
except:
  print("\nModel was not stored offline! It will be download soon\n")
  Model = VisionEncoderDecoderModel.from_pretrained(Model_Name) # , device_map = 	Execution_Device
  Model.save_pretrained(Base_Address + "Transformers/" + Model_Name.replace("/" , "_") + "/")

##Execution_Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##Model.gradient_checkpointing_enable()

# Processor = TrOCRProcessor.from_pretrained(Model_Name)
# Model = VisionEncoderDecoderModel.from_pretrained(Model_Name)

Model.config.max_length = 20
Model.config.max_new_tokens = 20


# Set special tokens used for creating the decoder_input_ids from the labels.
Model.config.decoder_start_token_id = Processor.tokenizer.cls_token_id
Model.config.pad_token_id = Processor.tokenizer.pad_token_id
# Set Correct vocab size.
Model.config.vocab_size = Model.config.decoder.vocab_size
Model.config.eos_token_id = Processor.tokenizer.sep_token_id


Model.config.max_length = 8
Model.config.early_stopping = False
Model.config.no_repeat_ngram_size = 3
Model.config.length_penalty = 2.0
Model.config.num_beams = 4

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#####
"""
if torch.cuda.device_count() > 1:
    print("Multiple Diveces Are Found, Parallelizing...")
    Model = torch.nn.DataParallel(Model).to(Execution_Device)
    #torch.multiprocessing.set_start_method("spawn", force = True)
    #
    #torch.dinamestributed.init_process_group(backend = "nccl")
    #Model = torch.nn.parallel.DistributedDataParallel(Model)

Model.to("cuda")
"""
try:
  Model.to("cuda")
except:
  pass
#####

print("\nExecution Device Is:", Execution_Device, "\n")
try:
	print("======================= =======================")
	try: # Parallelized With torch.nn.DataParallel
		print("Used Devices:", Model.device_ids)
	except: # Single Device
		print("Utilized Device:", Model.device)
	#print("Device Of Each Layers:", Model.hf_device_map)
	print("======================= =======================")
except:
	pass

####################### Inferencing TrOCR  ####################### #######################
# (Receiving Output of TrOCR and Evalute Metrics) 


"""
####################### Using Pipline #######################
from transformers import pipeline

pipe = pipeline("image-to-text", model = Model_Name)

Predictions = []
for i in range(Validation_Data.shape[0]):
  Predictions += [pipe(Validation_Data.iloc[i]["file_path"])[0]["generated_text"]]
"""

Reshape_Size = (150, 40)

####################### Using Model And Processor #######################
Predictions = []
for i in range(int(np.ceil(Validation_Data.shape[0] / Batch_Size))):
  Temp = np.array([np.array(Image.open(Validation_Data.iloc[j]["file_path"]).resize(Reshape_Size).convert("RGB")) for j in range(Batch_Size * i, Batch_Size * (i + 1)) if j < Validation_Data.shape[0]])
  #
  Temp = np.array([np.transpose(np.array([Temp[k - (Batch_Size * i), :, :, 2], Temp[k - (Batch_Size * i), :, :, 1], Temp[k - (Batch_Size * i), :, :, 0]]), axes = [1, 2, 0]) if "labeling-helper-source" in Validation_Data.iloc[k]["file_path"].lower() else Temp[k - (Batch_Size * i)] for k in range(Batch_Size * i, Batch_Size * (i + 1)) if k < Validation_Data.shape[0]])
  #
  Temp = Processor.batch_decode(Model.generate(torch.Tensor(np.array(Processor(Temp).pixel_values)).to(Execution_Device)), skip_special_tokens = True)
  Predictions += Temp

Predictions = [" ".join([j for j in Predictions[i] if j != " "]) for i in range(len(Predictions))]

# Storing_Path = "/content/Model/" + Model_Name[Model_Name.rfind("/") + 1:] + "_" + str(int(time.time())) + "_Inferencing" + "/"
Storing_Path = Base_Address + "Model/" + Model_Name[Model_Name.rfind("/") + 1:] + "_" + str(int(time.time())) + "_Inferencing" + "/"
print("Results Will Be Stored In:", Storing_Path)
if not os.path.isdir(Storing_Path): os.makedirs(Storing_Path)

# F = open(Storing_Path + "Labels.txt", "w")
F = open(Storing_Path + "Labels.txt", "w")
F.write("\n".join([j for j in Validation_Data.iloc[:, 1].to_numpy().tolist()]))
F.close()

# F = open(Storing_Path + "Predictions.txt", "w")
F = open(Storing_Path + "Predictions.txt", "w")
F.write("\n".join([j for j in Predictions]))
F.close()



####################### Save Log Of Parameters #######################
Parameters = {"Model_Name": Model_Name.replace("/" , "_")
              , "Epochs": Epochs
              , "Batch_Size": Batch_Size
              , "Execution_Device": Execution_Device.type
              , "Dataset Portion": Dataset_Portion
              , "Validation Size": Validation_Size
              , "Training Data Size": Train_Data.shape[0]
              , "Validation Data Size": Validation_Data.shape[0]
              # , "Test Data Size": len(test)
              , "Validation_Cumulation": Validation_Cumulation
              , "Using_Gray_Scale_Filters": Using_Gray_Scale_Filters
              # , "Concatenation_Length": Concatenation_Length
              , "Using_Synthetic_Dataset": Using_Synthetic_Dataset
              #, "Using_Synthetic_Dataset": round(Using_Synthetic_Dataset)
              , "Using_White_BackGround_Synthetic_Dataset": Using_White_BackGround_Synthetic_Dataset
              , "Using_Black_BackGround_Synthetic_Dataset": Using_Black_BackGround_Synthetic_Dataset
              , "Using_Augmented_Dataset": Using_Augmented_Dataset
              , "Using_Real_World_Dataset": Using_Real_World_Dataset
              , "Using_WhiteSpace": Using_WhiteSpace
              }

for i in range(len(Parameters.keys())):
  print(list(Parameters.items())[i])

F = open(Storing_Path + "Parameters.txt", "w")
# F.write(str(Parameters))
F.write(str(Parameters).replace(",", "\n"))
F.close()

# import json
# F = open("/content/Test.txt", "r")
# Parameters = json.loads(F.read().replace("\n", ",").replace("'", "\""))
# F.close()


Evaluation_Metrics = {}

try:
  CER_Metric = evaluate.load("cer")
  WER_Metric = evaluate.load("wer")
except:
  CER_Metric = load_metric("cer")
  WER_Metric = load_metric("wer")

Evaluation_Metrics["CER"] = CER_Metric.compute(predictions = Predictions, references = Validation_Data.iloc[:, 1].to_numpy().tolist())
Evaluation_Metrics["WER"] = WER_Metric.compute(predictions = Predictions, references = Validation_Data.iloc[:, 1].to_numpy().tolist())

# No Spaces
Evaluation_Metrics["CER_Not_Split"] = CER_Metric.compute(predictions = np.char.replace(np.array(Predictions), " ", "").tolist(), references = np.char.replace(Validation_Data.iloc[:, 1].to_numpy().astype("<U23"), " ", "").tolist())
Evaluation_Metrics["WER_Not_Split"] = WER_Metric.compute(predictions = np.char.replace(np.array(Predictions), " ", "").tolist(), references = np.char.replace(Validation_Data.iloc[:, 1].to_numpy().astype("<U23"), " ", "").tolist())

Evaluation_Metrics["Google_BLEU"] = evaluate.load("google_bleu").compute(predictions = Predictions, references = Validation_Data.iloc[:, 1].to_numpy().tolist())["google_bleu"]
Evaluation_Metrics["Google_BLEU_Not_Split"] = evaluate.load("google_bleu").compute(predictions = np.char.replace(np.array(Predictions), " ", "").tolist(), references = np.char.replace(Validation_Data.iloc[:, 1].to_numpy().astype("<U23"), " ", "").tolist())["google_bleu"]

"""
Temp_Validations = []
Temp_Predictions = []
for i in range(len(Predictions)):
  Temp_Validations += Processor.tokenizer(Validation_Data.iloc[i, 1])["input_ids"]
  Temp_Predictions += Processor.tokenizer(Predictions[i])["input_ids"]

Evaluation_Metrics["Accuracy"] = evaluate.load("accuracy").compute(predictions = Temp_Predictions, references = Temp_Validations)
"""

Evaluation_Metrics["Accuracy"] = 0.0
for i in range(len(Predictions)):
  if Predictions[i].lower() == Validation_Data.iloc[i, 1].lower():
    Evaluation_Metrics["Accuracy"] += 1
  #
Evaluation_Metrics["Accuracy"] /= len(Predictions)


F = open(Storing_Path + "Evaluation_Metrics.txt", "w")
# F.write(str(Parameters))
F.write(str(Evaluation_Metrics).replace(",", "\n"))
F.close()


### Loss Calculation ###
Model.config.decoder_start_token_id = Processor.tokenizer.cls_token_id
Model.config.pad_token_id = Processor.tokenizer.pad_token_id
# Set Correct vocab size.
Model.config.vocab_size = Model.config.decoder.vocab_size
Model.config.eos_token_id = Processor.tokenizer.sep_token_id

Evaluation_Metrics["Loss"] = []
with torch.no_grad():
  for i in range(Validation_Data.shape[0]):
    Temp = np.array(Image.open(Validation_Data.iloc[i]["file_path"]).resize(Reshape_Size).convert("RGB"))
    Temp = Processor(Temp, return_tensors = "pt")
    Temp = Model(**Temp.to(Execution_Device), labels = torch.Tensor([Processor.tokenizer(Validation_Data.iloc[i]["text"])["input_ids"]]).long().to(Execution_Device))
    Evaluation_Metrics["Loss"] += [float(Temp.loss)]

Evaluation_Metrics["Loss"] = np.mean(np.array(Evaluation_Metrics["Loss"]))
### ### ### ### ### ###

F = open(Storing_Path + "Evaluation_Metrics.txt", "w")
# F.write(str(Parameters))
F.write(str(Evaluation_Metrics).replace(",", "\n"))
F.close()

# import json
# F = open("/content/Test.txt", "r")
# Evaluation_Metrics = json.loads(F.read().replace("\n", ",").replace("'", "\""))
# F.close()

print(Evaluation_Metrics)




"""
if Using_Gray_Scale_Filters > 0:
    Temp = [Image.fromarray(Temp[j]).resize((600, 150)) for j in range(Temp.shape[0])]
    #
    for j in range(len(Temp)):
      if (Using_Augmented_Dataset == 1 or Using_Real_World_Dataset == 1) or "labeling-helper-source" in str(Validation_Data.iloc[j + (Batch_Size * i)]["file_path"]).lower():
        Temp[j] = ImageEnhance.Brightness(Temp[j]).enhance(60 / ImageStat.Stat(Temp[j]).mean[0])
        Temp[j] = ImageEnhance.Contrast(Temp[j]).enhance(60 / ImageStat.Stat(Temp[j]).stddev[0])
    #
    # 1: Combination Of Filters, 2: Concatenated Three GrayScale, 3: Concatenated fastNlMeansDenoising, 4: Concatenated Sobel Edge
    GrayScale_Image = [np.array(Temp[j].convert("L")) for j in range(len(Temp))] # Gray
    #
    Temp = np.array(Temp)
    if Using_Gray_Scale_Filters == 1 or Using_Gray_Scale_Filters == 2:
      for j in range(Temp.shape[0]):
        Temp[j, :, :, 0], Temp[j, :, :, 1], Temp[j, :, :, 2] = GrayScale_Image[j], GrayScale_Image[j], GrayScale_Image[j]
    #
    if Using_Gray_Scale_Filters > 0 or Using_Gray_Scale_Filters != 2:
      Blurred_Image = [cv2.GaussianBlur(GrayScale_Image[j], (5, 5), 0) for j in range(len(GrayScale_Image))]
    #
    if Using_Gray_Scale_Filters == 1 or Using_Gray_Scale_Filters == 3:
      Denoised_Image = [cv2.fastNlMeansDenoising(Blurred_Image[j], h = 10, templateWindowSize = 7, searchWindowSize = 21) for j in range(len(Blurred_Image))]
      Denoised_Image = [cv2.threshold(Denoised_Image[j], 64, 255, cv2.THRESH_BINARY)[1] for j in range(len(Denoised_Image))]
      Denoised_Image = [abs(Denoised_Image[j] - 255.0) if np.sum(Denoised_Image[j] < 64) > (0.65 * np.prod(list(Denoised_Image[j].shape))) else Denoised_Image[j] for j in range(len(Denoised_Image))]
      #
      for j in range(Temp.shape[0]):
        Temp[j, :, :, 0], Temp[j, :, :, 1], Temp[j, :, :, 2] = Denoised_Image[j], Denoised_Image[j], Denoised_Image[j]
    #
    if Using_Gray_Scale_Filters == 1 or Using_Gray_Scale_Filters == 4:
      Sobel_Edge_Image = [cv2.magnitude(cv2.Sobel(Blurred_Image[j], cv2.CV_64F, 1, 0, ksize = 3), cv2.Sobel(Blurred_Image[j], cv2.CV_64F, 0, 1, ksize = 3)) for j in range(len(Blurred_Image))]
      Sobel_Edge_Image = [((Sobel_Edge_Image[j] / np.max(Sobel_Edge_Image[j])) * 255).astype("uint8") for j in range(len(Sobel_Edge_Image))]
      Sobel_Edge_Image = [cv2.threshold(Sobel_Edge_Image[j], 50, 255, cv2.THRESH_BINARY)[1] for j in range(len(Sobel_Edge_Image))]
      Sobel_Edge_Image = [np.array(Image.fromarray(Sobel_Edge_Image[j]).filter(ImageEnhance.ImageFilter.SHARPEN).filter(ImageEnhance.ImageFilter.MedianFilter(3))) for j in range(len(Sobel_Edge_Image))] # .filter(ImageEnhance.ImageFilter.MaxFilter(3))
      Sobel_Edge_Image = [np.where(Sobel_Edge_Image[j] > 32, 0, 255) for j in range(len(Sobel_Edge_Image))]
      #
      for j in range(Temp.shape[0]):
        Temp[j, :, :, 0], Temp[j, :, :, 1], Temp[j, :, :, 2] = Sobel_Edge_Image[j], Sobel_Edge_Image[j], Sobel_Edge_Image[j]
      #
      if Using_Gray_Scale_Filters == 1:
        for j in range(Temp.shape[0]):
          Temp[j, :, :, 0], Temp[j, :, :, 1], Temp[j, :, :, 2] = GrayScale_Image[j], Denoised_Image[j], Sobel_Edge_Image[j]
      #
      del Blurred_Image, Denoised_Image, Sobel_Edge_Image
"""


print()


