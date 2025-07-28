import numpy as np
import os
import matplotlib.pyplot as plt

from IPython.display import clear_output

import json

clear_output()

####################### Global Parameters ####################### #######################
Base_Address = "../"
Data_Address = "../Data/"

Model_Name = "microsoft/trocr-small-printed"
#Model_Name = "microsoft/trocr-base-printed"
#Model_Name = "microsoft/trocr-large-printed"

####################### History Plotting ####################### #######################


# trocr-small-printed_1739107801_Fine-Tuning
Models_Name = [name for name in os.listdir(Base_Address + "Model/") if (("trocr" in name.lower() and "fine-tun" in name.lower()) and (os.path.isfile(Base_Address + "Model/" + name + "/trainer_state.json")))]
Models_Name.sort()

# Plot_Metric = "CER" # "WER"
Plot_Metric = "loss"

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
    #
    plt.plot(Temp[:, 0], Temp[:, 1], linestyle = "-", linewidth = 3.1, label = "Training")
  except:
    print("The Metrics Does Not Exist In The Training History!!")
  #
  try:
    Temp = [[History["log_history"][i]["epoch"], History["log_history"][i]["eval_" + Plot_Metric]] for i in range(len(History["log_history"])) if "eval_" + Plot_Metric in History["log_history"][i].keys()]
    Temp = np.array(Temp)
    # 
    plt.plot(Temp[:, 0], Temp[:, 1], linestyle = "-", linewidth = 3.1, label = "Validation")
  except:
    print("The Metrics Does Not Exist In The Validation History!!")

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
