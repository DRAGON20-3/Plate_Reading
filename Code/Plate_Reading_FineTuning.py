import numpy as np
import pandas as pd
from PIL import Image, ImageStat, ImageEnhance
import os
import matplotlib.pyplot as plt
import time
import shutil

from IPython.display import clear_output

import torch
from torch.utils.data import Dataset #, DataLoader
from transformers import VisionEncoderDecoderModel, TrOCRProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
import cv2
import json

os.system("pip install gdown --upgrade") # !pip install gdown --upgrade
clear_output()
import gdown

try:
  import accelerate
except:
  os.system("pip install accelerate") # !pip install accelerate
  clear_output()
  import accelerate

try:
  import evaluate
except:
  os.system("pip install evaluate")
  clear_output()
  import evaluate

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

#os.environ["CUDA_VISIBLE_DEVICES"] = "1" # In Order To Select GPU

Execution_Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Base_Address = "../"
Data_Address = "../Data/"

Model_Name = "microsoft/trocr-small-printed"
#Model_Name = "microsoft/trocr-base-printed"
#Model_Name = "microsoft/trocr-large-printed"

Using_Synthetic_Dataset = 1
#Using_White_BackGround_Synthetic_Dataset = 0
#Using_Black_BackGround_Synthetic_Dataset = 0
Using_Augmented_Dataset = 0
Using_Real_World_Dataset = 0


Dataset_Portion = 1.0 # between 0.0 and 1.0 to choose a portion of Trainin Data in order to tackle low RAM memory for loading data
Validation_Size = 0.2 # 0.23 means 23%


Validation_Cumulation = 2 # 0: Just Using the Evaludation Set Of Real Data, 1: Concatenate The Evaludation Set Of Real Data with Validation Set of The Mentioned Section, 2: Just Validation Set Of The Mentioned Section
Evaluation_Section = 1 # 1: 500 Dubizzle, (DO NOT CHANGE OR REMOVE THIS PARAMETER)

Using_Gray_Scale_Filters = 0 # Using Grayscale and Filters Over The Image Instead Of RGB Image; 1: Combination Of Filters, 2: Concatenated Three GrayScale, 3: Concatenated fastNlMeansDenoising, 4: Concatenated Sobel Edge
Using_WhiteSpace = 0 # Using WhiteSpace In Tokenization Between Each Letter (1: "1 3 4 B", 0: "134B")
Style_Shifting = "1" # Changing The Style Of Synthetic Data To Be Similar To Real Data (Like Augmentations Techniques); 1: All, 2: BackGround, 3: Brightness, 4: Stratching, 5: Rotation, 6: Noise; Could be Merge like "23" Means both Background and Brightness

Epochs = 10 
Batch_Size = 32


####################### Data Loading ####################### #######################

if Using_Synthetic_Dataset == 1: # or Using_Augmented_Dataset == 1:
  if not os.path.isdir(Data_Address + "Synthetic_Data/Data/"):
    if not os.path.isfile(Data_Address + "Synthetic_Data/Synthetic_Data.zip"):
      gdown.download(id = "1VBYWCgQSgChnfQTI0v4jU_pf0-QJ-LzE", output = Data_Address + "Synthetic_Data/Synthetic_Data.zip") # https://drive.google.com/file/d/1VBYWCgQSgChnfQTI0v4jU_pf0-QJ-LzE
    shutil.unpack_archive(Data_Address + "Synthetic_Data/Synthetic_Data.zip", Data_Address + "Synthetic_Data/")
    
if Using_Augmented_Dataset == 1: # or Using_Augmented_Dataset == 1:
  if not os.path.isdir(Data_Address + "Augmented_Data/Data/"):
    if not os.path.isfile(Data_Address + "Augmented_Data/Augmented_Data.zip"):
      gdown.download(id = "12p7wLegr8q_nXT3UxUHv7-PbmwE5hnHJ", output = Data_Address + "Augmented_Data/Augmented_Data.zip") # https://drive.google.com/file/d/12p7wLegr8q_nXT3UxUHv7-PbmwE5hnHJ
    shutil.unpack_archive(Data_Address + "Augmented_Data/Augmented_Data.zip", Data_Address + "Augmented_Data/")
      
    
Data_Train = pd.DataFrame([])
Data_Test = pd.DataFrame([])

if Using_Synthetic_Dataset == 1:
  Temp = pd.read_csv(Data_Address + "Synthetic_Data/Data.csv")
  Temp["file_path"] = np.char.replace(Temp["file_path"].to_numpy().astype("<U230"), "./", Data_Address + "Synthetic_Data/")
  Temp["bbox_path"] = np.char.replace(Temp["bbox_path"].to_numpy().astype("<U230"), "./", Data_Address + "Synthetic_Data/")
  #
  np.random.seed(23)
  Temp_2 = np.random.choice(Temp.shape[0], size = Temp.shape[0], replace = False)
  # Data = pd.concat([Data, Temp], ignore_index = True)
  Data_Train = pd.concat([Data_Train, Temp.iloc[Temp_2[:int(-1 * (Validation_Size * Temp_2.shape[0]))]] ], ignore_index = True)
  Data_Test = pd.concat([Data_Test, Temp.iloc[Temp_2[int(-1 * (Validation_Size * Temp_2.shape[0])):]] ], ignore_index = True)

if Using_Augmented_Dataset == 1:
  Temp = pd.read_csv(Data_Address + "Augmented_Data/Data.csv")
  Temp["file_path"] = np.char.replace(Temp["file_path"].to_numpy().astype("<U230"), "./", Data_Address + "Augmented_Data/")
  Temp["bbox_path"] = np.char.replace(Temp["bbox_path"].to_numpy().astype("<U230"), "./", Data_Address + "Augmented_Data/")
  #
  np.random.seed(23)
  Temp_2 = np.random.choice(Temp.shape[0], size = Temp.shape[0], replace = False)
  # Data = pd.concat([Data, Temp], ignore_index = True)
  Data_Train = pd.concat([Data_Train, Temp.iloc[Temp_2[:int(-1 * (Validation_Size * Temp_2.shape[0]))]] ], ignore_index = True)
  Data_Test = pd.concat([Data_Test, Temp.iloc[Temp_2[int(-1 * (Validation_Size * Temp_2.shape[0])):]] ], ignore_index = True)
  #

if Using_Real_World_Dataset == 1:
  Temp = pd.read_csv(Data_Address + "Real_Data/Train.csv")
  Temp["file_path"] = np.char.replace(Temp["file_path"].to_numpy().astype("<U230"), "./", Data_Address + "Real_Data/")
  Temp["bbox_path"] = np.char.replace(Temp["bbox_path"].to_numpy().astype("<U230"), "./", Data_Address + "Real_Data/")
  #
  Temp_2 = pd.read_csv(Data_Address + "Real_Data/Validation.csv")
  Temp_2["file_path"] = np.char.replace(Temp_2["file_path"].to_numpy().astype("<U230"), "./", Data_Address + "Real_Data/")
  Temp_2["bbox_path"] = np.char.replace(Temp_2["bbox_path"].to_numpy().astype("<U230"), "./", Data_Address + "Real_Data/")
  #
  Data_Train = pd.concat([Data_Train, Temp], ignore_index = True)
  Data_Test = pd.concat([Data_Test, Temp_2], ignore_index = True)
  #


if "Temp" in globals(): del Temp
if "Temp_2" in globals(): del Temp_2


Evaluation_Data = np.array([])
if Validation_Cumulation != 2:
  Evaluation_Data = pd.read_csv(Data_Address + "Real_Data/Validation.csv")
  Evaluation_Data["file_path"] = np.char.replace(Evaluation_Data["file_path"].to_numpy().astype("<U230"), "./", Data_Address + "Real_Data/")
  Evaluation_Data["bbox_path"] = np.char.replace(Evaluation_Data["bbox_path"].to_numpy().astype("<U230"), "./", Data_Address + "Real_Data/")


if "Dataset_Portion" in globals() and Dataset_Portion < 1.0:
  np.random.seed(23)
  Temp = np.random.choice(Data_Train.shape[0], size = int(Dataset_Portion * Data_Train.shape[0]), replace = False)
  Data_Train = Data_Train.iloc[Temp]
  del Temp


######### Remove Non-English Letters
Data_Train.iloc[:, 1] = np.char.decode(np.char.encode(Data_Train.iloc[:, 1].to_numpy().astype(np.str_), encoding = "ascii", errors = "ignore"))
Data_Train.iloc[:, 1] = np.char.replace(np.char.strip(Data_Train.iloc[:, 1].to_numpy().astype(np.str_)), "  ", " ")

if len(Data_Test) > 0:
  Data_Test.iloc[:, 1] = np.char.decode(np.char.encode(Data_Test.iloc[:, 1].to_numpy().astype(np.str_), encoding = "ascii", errors = "ignore"))
  Data_Test.iloc[:, 1] = np.char.replace(np.char.strip(Data_Test.iloc[:, 1].to_numpy().astype(np.str_)), "  ", " ")

if Validation_Cumulation != 2:
  Evaluation_Data.iloc[:, 1] = np.char.decode(np.char.encode(Evaluation_Data.iloc[:, 1].to_numpy().astype(np.str_), encoding = "ascii", errors = "ignore"))
  Evaluation_Data.iloc[:, 1] = np.char.replace(np.char.strip(Evaluation_Data.iloc[:, 1].to_numpy().astype(np.str_)), "  ", " ")


if Validation_Cumulation == 0:
  Data_Validation = Evaluation_Data.copy()
elif Validation_Cumulation == 1:
  Data_Validation = pd.concat([Evaluation_Data, Data_Test], ignore_index = True)
elif Validation_Cumulation == 2: # Inferencing
  Data_Validation = Data_Test.copy()

del Data_Test

### Remove F As it may be in old plates with Green background color
Data_Train["text"] = np.char.replace(np.char.upper(Data_Train["text"].to_numpy().astype("<U23")), "F", "")
Data_Validation["text"] = np.char.replace(np.char.upper(Data_Validation["text"].to_numpy().astype("<U23")), "F", "")



print("Data_Train Shape:", Data_Train.shape)
print("Data_Validation Shape:", Data_Validation.shape)
print("Evaluation_Data Shape:", Evaluation_Data.shape)


print()

####################### Loading TrOCR Model ####################### #######################
##torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

if os.path.exists(Base_Address + "Model/" + Model_Name + "/"):
  F = open(Base_Address + "Model/" + Model_Name + "/Parameters.txt", "r")
  Temp = json.loads(F.read().replace("\n", ",").replace("'", "\""))
  F.close()
  # 
  Temp = [[Base_Address + "Model/" + Model_Name + "/" + name, os.path.getmtime(Base_Address + "Model/" + Model_Name + "/" + name)] for name in os.listdir(Base_Address + "Model/" + Model_Name + "/") if ("checkpoint" in name.lower() and os.path.isdir(Base_Address + "Model/" + Model_Name + "/" + name))]
  Temp = np.array(Temp)
  if len(Temp) == 0 : 
    print("No CheckPoint Is Available!!")
    Temp = Model_Name
    #quit() # print(SKIP)
  else:
    Temp = Temp[np.argmax(Temp[:, 1].astype(np.float64)), 0]
  #
  Model = VisionEncoderDecoderModel.from_pretrained(Temp) # , device_map = Execution_Device
elif not os.path.exists(Base_Address + "Model/" + Model_Name + "/") or "Model" not in globals():
  try:
    Model = VisionEncoderDecoderModel.from_pretrained(Base_Address + "Transformers/" + Model_Name.replace("/" , "_") + "/") # , device_map = Execution_Device
  except:
	  print("\nModel was not stored offline! It will be download soon\n")
	  Model = VisionEncoderDecoderModel.from_pretrained(Model_Name) # , device_map = Execution_Device
	  Model.save_pretrained(Base_Address + "Transformers/" + Model_Name.replace("/" , "_") + "/")

try:
  Processor = TrOCRProcessor.from_pretrained(Base_Address + "Transformers/" + Model_Name.replace("/" , "_") + "/Processor/")
except:
  Processor = TrOCRProcessor.from_pretrained(Model_Name)
  Processor.save_pretrained(Base_Address + "Transformers/" + Model_Name.replace("/" , "_") + "/Processor/")

Model.config.max_length = 20
Model.config.max_new_tokens = 20

# Set special tokens used for creating the decoder_input_ids from the labels.
Model.config.decoder_start_token_id = Processor.tokenizer.cls_token_id
Model.config.pad_token_id = Processor.tokenizer.pad_token_id
# Set Correct vocab size.
Model.config.vocab_size = Model.config.decoder.vocab_size
Model.config.eos_token_id = Processor.tokenizer.sep_token_id


Model.config.early_stopping = False
#Model.config.no_repeat_ngram_size = 3
#Model.config.length_penalty = 2.0
#Model.config.num_beams = 4

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
  print(AA)
  Model.to("cuda")
except:
  pass

print("\nExecution Device Is:", Execution_Device, "\n")
try:
	print("======================= =======================")
	try: # Parallelized With torch.nn.DataParallel
		print("Used Devices:", Model.device_ids)
	except: # Single Device
		print("Utilized Device:", Model.device)
	print("======================= =======================")
except:
	pass

####################### FineTuning TrOCR ####################### #######################


class CustomOCRDataset(Dataset):
  def __init__(self, Base_Address, df, Processor, max_target_length = 128):
    self.root_dir = Base_Address
    self.df = df
    self.Processor = Processor
    self.max_target_length = max_target_length


  def __len__(self):
    return len(self.df)


  def __getitem__(self, idx):
    
    file_path = self.df["file_path"].iloc[idx] # The image file name.

    text = self.df["text"].iloc[idx] # The text (label).

    image = Image.open(self.root_dir + file_path).convert("RGB") # Read the image, apply augmentations, and get the transformed pixels.
    #
    if "labeling-helper-source" in str(self.root_dir + file_path).lower():
      image = np.array(image)
      image = np.array([image[:, :, 2], image[:, :, 1], image[:, :, 0]])
      image = np.transpose(image, axes = [1, 2, 0])
      image = Image.fromarray(image)
    #
    Resize_Shape = (160, 40)
    ####################### Style Shifting #######################
    if "0" not in Style_Shifting and (trainer.control.should_evaluate == False or self.df.shape[0] == Data_Validation.df.shape[0]) and (Using_Synthetic_Dataset == 1): # (Using_Synthetic_Dataset == 1 or Using_White_BackGround_Synthetic_Dataset == 1 or Using_Black_BackGround_Synthetic_Dataset == 1)
      # 1: All, 2: BackGround, 3: Brightness, 4: Stratching, 5: Rotation, 6: Noise
      #
      Temp_Image = np.array(image)
      Temp_Image[:, :, 0] = np.where(Temp_Image[:, :, 0] > 160, 255, Temp_Image[:, :, 0])
      Temp_Image[:, :, 1] = np.where(Temp_Image[:, :, 1] > 144, 150, Temp_Image[:, :, 1])
      image = Image.fromarray(Temp_Image.astype(np.uint8)) # .resize((160, 40))
      #
      Padding_Value = [15, 10, 15, 10] # X1, Y1, X2, Y2
      #
      if "1" in Style_Shifting or "2" in Style_Shifting:
        #
        Temp_Padded = np.pad(np.array(image).astype(np.float16), ([Padding_Value[0], Padding_Value[2]], [Padding_Value[1], Padding_Value[3]], [0, 0])
                            , constant_values = -255) # tuple(np.random.randint(10, 20, size = 4).reshape(2, 2).tolist() + [[0, 0]])

        if np.random.rand() > 0.4:
          Temp_Colors = np.tile(np.random.randint(32, 255), 3).tolist()
        else:
          Temp_Colors = np.random.randint(0, 48, size = 3).astype(np.float32)
          if np.random.rand() > 0.3: # Dark Color
            Temp_Colors[np.random.choice(np.arange(len(Temp_Colors)))] *= 5
          else: # Light Color
            Temp_Colors[np.random.choice(np.arange(len(Temp_Colors)), size = 2)] *= [5.0, 2.5]
          #
          Temp_Colors = Temp_Colors.tolist()

        Temp_Padded[:, :, 0] = np.where(Temp_Padded[:, :, 0] == -255.0, Temp_Colors[0], Temp_Padded[:, :, 0])
        Temp_Padded[:, :, 1] = np.where(Temp_Padded[:, :, 1] == -255.0, Temp_Colors[1], Temp_Padded[:, :, 1])
        Temp_Padded[:, :, 2] = np.where(Temp_Padded[:, :, 2] == -255.0, Temp_Colors[2], Temp_Padded[:, :, 2])
        Temp_Padded = Temp_Padded.astype(np.uint8)

        image = Image.fromarray(Temp_Padded.astype(np.uint8))

      #
      if "1" in Style_Shifting or "3" in Style_Shifting:
        #
        Temp_Adjusted = image # Image.fromarray(Temp_Padded)
        Temp_Adjusted = Temp_Adjusted.filter(ImageEnhance.ImageFilter.EDGE_ENHANCE)
        Temp_Adjusted = ImageEnhance.Brightness(Temp_Adjusted).enhance(np.random.randint(40, 190) / ImageStat.Stat(Temp_Adjusted).mean[0]) # Brightness Adjustment
        Temp_Adjusted = ImageEnhance.Contrast(Temp_Adjusted).enhance(np.random.randint(60, 100) / ImageStat.Stat(Temp_Adjusted).stddev[0]) # Contrast Adjustment
        #
        if np.random.rand() < 0.5:
          Temp_Adjusted = Temp_Adjusted.filter(ImageEnhance.ImageFilter.GaussianBlur(0.5))
        else:
          Temp_Coefficent = np.random.randint(60, 80) * 0.01
          Temp_Adjusted = Temp_Adjusted.resize((int(Temp_Coefficent * Resize_Shape[0]), int(Temp_Coefficent * Resize_Shape[1]))).resize(Resize_Shape)
        Temp_Adjusted = np.array(Temp_Adjusted)
        #
        image = Image.fromarray(Temp_Adjusted.astype(np.uint8))
      #
      if ("1" in Style_Shifting and np.random.rand() > 0.20) or "4" in Style_Shifting:
        #
        image = np.array(image)
        if "1" not in Style_Shifting: # Need To Pad
          image = np.pad(image, ([Padding_Value[0], Padding_Value[2]], [Padding_Value[1], Padding_Value[3]], [0, 0]), mode = "edge")
        #
        Source_Positions = np.array([[Padding_Value[1], Padding_Value[0]]
                          , [Temp_Image.shape[1] + Padding_Value[1], Padding_Value[0]]
                          , [Temp_Image.shape[1] + Padding_Value[1], Temp_Image.shape[0] + Padding_Value[0]]
                          , [Padding_Value[1], Temp_Image.shape[0] + Padding_Value[0]]]).astype(np.float32) # Top Left, Top Right, Bottom Right, Bottom Left

        #
        Target_Positions = np.concatenate((np.random.randint(int(1.5 * Padding_Value[1]), size = (4, 1)), np.random.randint(int(1.5 * Padding_Value[0]), size = (4, 1))) , axis = 1).astype(np.float32)
        # 
        Target_Positions = np.abs(np.subtract(np.array([[0, 0]
                                                    , [image.shape[1], 0]
                                                    , [image.shape[1], image.shape[0]]
                                                    , [0, image.shape[0]]])
                                            , Target_Positions)).astype(np.float32)
        #
        Temp_Overflow = np.multiply((np.random.rand(8) > 0.7), np.array([-1, -1, 1, -1, 1, 1, -1, 1])).reshape(4, 2)
        Target_Positions = np.add(5.0 * Temp_Overflow, Target_Positions).astype(np.float32)
        #
        Temp_Shifted = cv2.warpPerspective(image, cv2.getPerspectiveTransform(Source_Positions, Target_Positions), (image.shape[1], image.shape[0]))
        #
        image = Image.fromarray(Temp_Shifted.astype(np.uint8))

      elif "1" in Style_Shifting or "5" in Style_Shifting:
        image = image.resize((2 * image.width, 2 * image.height))
        #
        Temp_Rotate = np.pad(np.array(image), ((15, 15), (15, 15), (0, 0)), mode = "edge")
        Temp_Rotate = Image.fromarray(Temp_Rotate).rotate(np.random.randint(-10, 10))
        Temp_Rotate = np.array(Temp_Rotate)[20:-20, 20:-20]
        #
        image = Image.fromarray(Temp_Rotate.astype(np.uint8))
      #
      if "1" in Style_Shifting or "6" in Style_Shifting:
        Temp_Noise = np.array(image)
        #
        if np.random.rand() < 0.6:
          Temp_Noise = np.add(Temp_Noise, np.random.poisson(lam = 11, size = Temp_Noise.shape))
        else:
          if np.random.rand() > 0.5:
            Temp_Noise = np.add(Temp_Noise, np.sum(np.random.multivariate_normal(np.ones((2)), np.eye((2)) * 11, size = Temp_Noise.shape), axis = -1))
          else:
            Temp_Noise = np.add(Temp_Noise, np.random.normal(0, 11, size = Temp_Noise.shape))
        #
        Temp_Noise = np.where(Temp_Noise > 255, 255, Temp_Noise)
        Temp_Noise = np.where(Temp_Noise < 0, 0, Temp_Noise)
        Temp_Noise = Temp_Noise.astype(np.uint8)
        #
        image = Image.fromarray(Temp_Noise.astype(np.uint8))
      #
      image = image.resize(Resize_Shape)
      #
    ####################### #######################
    if Using_Gray_Scale_Filters > 0:
      Temp = image.resize((600, 150))
      #
      if (Using_Augmented_Dataset == 1 or Using_Real_World_Dataset == 1) or "labeling-helper-source" in str(self.root_dir + file_path).lower():
        Temp = ImageEnhance.Brightness(Temp).enhance(60 / ImageStat.Stat(Temp).mean[0])
        Temp = ImageEnhance.Contrast(Temp).enhance(60 / ImageStat.Stat(Temp).stddev[0])
      #
      # 1: Combination Of Filters, 2: Concatenated Three GrayScale, 3: Concatenated fastNlMeansDenoising, 4: Concatenated Sobel Edge
      GrayScale_Image = np.array(Temp.convert("L")) # Gray
      #
      Temp = np.array(Temp)
      if Using_Gray_Scale_Filters == 1 or Using_Gray_Scale_Filters == 2:
        Temp[:, :, 0], Temp[:, :, 1], Temp[:, :, 2] = GrayScale_Image, GrayScale_Image, GrayScale_Image
      #
      if Using_Gray_Scale_Filters > 0 or Using_Gray_Scale_Filters != 2:
        Blurred_Image = cv2.GaussianBlur(GrayScale_Image, (5, 5), 0)
      #
      if Using_Gray_Scale_Filters == 1 or Using_Gray_Scale_Filters == 3:
        Denoised_Image = cv2.fastNlMeansDenoising(Blurred_Image, h = 10, templateWindowSize = 7, searchWindowSize = 21)
        Denoised_Image = cv2.threshold(Denoised_Image, 64, 255, cv2.THRESH_BINARY)[1]
        Denoised_Image = abs(Denoised_Image - 255.0) if np.sum(Denoised_Image < 64) > (0.65 * np.prod(list(Denoised_Image.shape))) else Denoised_Image
        #
        Temp[:, :, 0], Temp[:, :, 1], Temp[:, :, 2] = Denoised_Image, Denoised_Image, Denoised_Image
      #
      if Using_Gray_Scale_Filters == 1 or Using_Gray_Scale_Filters == 4:
        Sobel_Edge_Image = cv2.magnitude(cv2.Sobel(Blurred_Image, cv2.CV_64F, 1, 0, ksize = 3), cv2.Sobel(Blurred_Image, cv2.CV_64F, 0, 1, ksize = 3))
        Sobel_Edge_Image = ((Sobel_Edge_Image / np.max(Sobel_Edge_Image)) * 255).astype("uint8")
        Sobel_Edge_Image = cv2.threshold(Sobel_Edge_Image, 50, 255, cv2.THRESH_BINARY)[1]
        Sobel_Edge_Image = np.array(Image.fromarray(Sobel_Edge_Image).filter(ImageEnhance.ImageFilter.SHARPEN).filter(ImageEnhance.ImageFilter.MedianFilter(3))) # .filter(ImageEnhance.ImageFilter.MaxFilter(3))
        Sobel_Edge_Image = np.where(Sobel_Edge_Image > 32, 0, 255)
        #
        Temp[:, :, 0], Temp[:, :, 1], Temp[:, :, 2] = Sobel_Edge_Image, Sobel_Edge_Image, Sobel_Edge_Image
        #
      if Using_Gray_Scale_Filters == 1:
        Temp[:, :, 0], Temp[:, :, 1], Temp[:, :, 2] = GrayScale_Image, Denoised_Image, Sobel_Edge_Image
        #
      image = Image.fromarray(Temp)
      #
      # del Blurred_Image, Denoised_Image, Sobel_Edge_Image
      if "Blurred_Image" in globals(): del Blurred_Image
      if "Denoised_Image" in globals(): del Denoised_Image
      if "Sobel_Edge_Image" in globals(): del Sobel_Edge_Image
      # image = Image.fromarray(Temp).resize((np.array(image).shape[1], np.array(image).shape[0]))
      #
      del GrayScale_Image
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
    #
    pixel_values = self.Processor(image, return_tensors="pt").pixel_values # image = train_transforms(image) # Augmentation

    # Pass the text through the tokenizer and get the labels, i.e. tokenized labels.
    labels = self.Processor.tokenizer(
        text,
        padding="max_length",
        max_length=self.max_target_length
    ).input_ids

    labels = [label if label != self.Processor.tokenizer.pad_token_id else -100 for label in labels] # Using -100 as the padding token.
    encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
    return encoding



Data_Train = CustomOCRDataset(
  Base_Address = "",
  df = Data_Train,
  Processor = Processor
)


Data_Validation = CustomOCRDataset(
  Base_Address = "",
  df = Data_Validation,
  Processor = Processor
)



print("Data Is Ready")

#######################
# Model.config.encoder.image_size = Model.config.encoder.hidden_size
# Processor.image_processor.size["width"] = Model.config.encoder.hidden_size
# Processor.image_processor.size["height"] = Model.config.encoder.hidden_size
#######################


print("Model Is Configured And Ready")

# Optimizer = optim.AdamW(Model.parameters(), lr = TrainingConfig.LEARNING_RATE, weight_decay = 0.0005)
# Optimizer = optim.Adam(Model.parameters(), lr = 0.00005, weight_decay = 0.0005)

try:
  CER_Metric = evaluate.load("cer")
  WER_Metric = evaluate.load("wer")
except:
  CER_Metric = load_metric("cer")
  WER_Metric = load_metric("wer")

def Compute_Error_Rates(pred):
  labels_ids = pred.label_ids
  pred_ids = pred.predictions

  pred_ids[pred_ids == -100] = Processor.tokenizer.pad_token_id
  pred_str = Processor.batch_decode(pred_ids, skip_special_tokens=True)
  labels_ids[labels_ids == -100] = Processor.tokenizer.pad_token_id
  label_str = Processor.batch_decode(labels_ids, skip_special_tokens=True)

  #######
  F = open(Storing_Path + "Labels.txt", "w")
  F.write("\n".join([j for j in label_str]))
  F.close()

  F = open(Storing_Path + "Predictions.txt", "w")
  F.write("\n".join([j for j in pred_str]))
  F.close()
  #######

  CER = CER_Metric.compute(predictions = pred_str, references = label_str)
  WER = WER_Metric.compute(predictions = pred_str, references = label_str)
  return {"CER": CER, "WER": WER}


Storing_Path = Base_Address + "Model/" + Model_Name[Model_Name.rfind("/") + 1:] + "_" + str(int(time.time())) + "_Fine-Tuning" + "/"


####################### Save Log Of Parameters #######################
Parameters = {"Model_Name": Model_Name.replace("/" , "_")
              , "Epochs": Epochs
              , "Batch_Size": Batch_Size
              , "Execution_Device": Execution_Device.type
              , "Dataset Portion": Dataset_Portion
              , "Validation Size": Validation_Size
              , "Training Data Size": Data_Train.df.shape[0]
              , "Validation Data Size": Data_Validation.df.shape[0]
              # , "Test Data Size": len(test)
              , "Validation_Cumulation": Validation_Cumulation
              , "Evaluation_Section": Evaluation_Section
              , "Using_Gray_Scale_Filters": Using_Gray_Scale_Filters
              , "Style_Shifting": Style_Shifting
              , "Using_Synthetic_Dataset": Using_Synthetic_Dataset
              #, "Using_Synthetic_Dataset": round(Using_Synthetic_Dataset)
              , "Using_Augmented_Dataset": Using_Augmented_Dataset
              , "Using_Real_World_Dataset": Using_Real_World_Dataset
              , "Using_WhiteSpace": Using_WhiteSpace
              }

for i in range(len(Parameters.keys())):
  print(list(Parameters.items())[i])


####################### Continue Existed Run #######################
if not os.path.isdir(Base_Address + "Model/"): os.makedirs(Base_Address + "Model/")
Temp_Folders = [Base_Address + "Model/" + name for name in os.listdir(Base_Address + "Model/") if ("Fine-Tuning".lower() in name.lower() and os.path.isfile(Base_Address + "Model/" + name + "/Parameters.txt"))]
for i in range(len(Temp_Folders)):
  F = open(Temp_Folders[i] + "/Parameters.txt", "r")
  Temp = json.loads(F.read().replace("\n", ",").replace("'", "\""))
  F.close()
  #
  if (False not in [Temp[i] == Parameters[i] for i in list(Temp.keys()) if i not in ["Epochs", "Execution_Device"]]) and list(Temp.keys()) == list(Parameters.keys()): # Model_Name
    Temp_Storing_Path = Storing_Path
    Storing_Path = Temp_Folders[i] + "/"
    print("----------------------- -----------------------")
    print("Similar Execution Were Found")
    print("----------------------- -----------------------")
    break
    ####################### #######################
    # OverWrite To Make Sure Some Details Like Epochs Are The Same
    #
    F = open(Storing_Path + "Parameters.txt", "w")
    F.write(str(Parameters).replace(",", "\n"))

####################### ####################### #######################



print("Results Will Be Stored In:", Storing_Path)
if not os.path.isdir(Storing_Path): os.makedirs(Storing_Path)


Train_Parameters = Seq2SeqTrainingArguments(
  output_dir = Storing_Path,
  predict_with_generate = True,
  eval_strategy = "epoch",
  per_device_train_batch_size = Batch_Size, # 32
  per_device_eval_batch_size = 16,
  fp16 = True,
  logging_strategy = "epoch",
  save_strategy = "epoch",
  save_total_limit = 1,
  report_to = "tensorboard",
  num_train_epochs = Epochs # 10
  #, dataloader_num_workers = 32
  #, split_batches = True
  , remove_unused_columns = False
  , dataloader_pin_memory = False
)

print("Execution Devices Count:", Train_Parameters.n_gpu)
print("Parameters Were Set")

trainer = Seq2SeqTrainer(
  model = Model,
  tokenizer = Processor.feature_extractor,
  args = Train_Parameters,
  compute_metrics = Compute_Error_Rates,
  train_dataset = Data_Train,
  eval_dataset = Data_Validation,
  data_collator = default_data_collator
)


print("Starting Fine-Tuning", Model_Name, "With", Batch_Size, "Batches And", Epochs, "Epochs ...")
print(Train_Parameters.output_dir)
Start_Time = time.time()
#
try:
  time.sleep(1.15)
  trainer.train(resume_from_checkpoint = True)
  del Temp_Storing_Path
except:
  #
  if "Temp_Storing_Path" in globals():
    Storing_Path = Temp_Storing_Path
    del Temp_Storing_Path


  ####################### Save Log Of Parameters #######################
  Parameters = {"Model_Name": Model_Name.replace("/" , "_")
                , "Epochs": Epochs
                , "Batch_Size": Batch_Size
                , "Execution_Device": Execution_Device.type
                , "Dataset Portion": Dataset_Portion
                , "Validation Size": Validation_Size
                , "Training Data Size": Data_Train.df.shape[0]
                , "Validation Data Size": Data_Validation.df.shape[0]
                # , "Test Data Size": len(test)
                , "Validation_Cumulation": Validation_Cumulation
                , "Evaluation_Section": Evaluation_Section
                , "Using_Gray_Scale_Filters": Using_Gray_Scale_Filters
                , "Style_Shifting": Style_Shifting
                , "Using_Synthetic_Dataset": Using_Synthetic_Dataset
                #, "Using_Synthetic_Dataset": round(Using_Synthetic_Dataset)
                , "Using_Augmented_Dataset": Using_Augmented_Dataset
                , "Using_Real_World_Dataset": Using_Real_World_Dataset
                , "Using_WhiteSpace": Using_WhiteSpace
                }

  for i in range(len(Parameters.keys())):
    print(list(Parameters.items())[i])
  #
  if not os.path.isdir(Storing_Path): os.makedirs(Storing_Path)
  #
  F = open(Storing_Path + "Parameters.txt", "w")
  # F.write(str(Parameters))
  F.write(str(Parameters).replace(",", "\n"))
  F.close()

  ####################### ####################### #######################
  #
  trainer.train()

trainer.save_state()
#
print("Training Progress Takes ", int(time.time() - Start_Time), "Seconds")


F = open(Storing_Path + "Training_Arguments.txt", "w")
F.write(str(trainer.args))
# F.write(str(trainer.args).replace(",", "\n"))
F.close()

History = trainer



print()


