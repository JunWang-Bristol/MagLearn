import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import NW_LSTM
import NN_DataLoader

import Maglib
import linear_std



def MagLoss(
    B_waveform,
    Temp,
    Freq,
    model_saved_name="model.ckpt",
    dataset_path=r"data\std_dataset",
    function_use="valid"
    ):

    std_b = linear_std.linear_std()
    std_freq = linear_std.linear_std()
    std_temp = linear_std.linear_std()
    std_loss = linear_std.linear_std()

    std_b.load(dataset_path+r"\std_b.stdd")
    std_freq.load(dataset_path+r"\std_freq.stdd")
    std_temp.load(dataset_path+r"\std_temp.stdd")
    std_loss.load(dataset_path+r"\std_loss.stdd")


    # Check if CUDA is available and if so, set the device to GPU
    device = torch.device("cpu")
    #print("Device using ", device)

    # Instantiate the model with appropriate dimensions
    model = NW_LSTM.get_global_model().to(device)

    # load model from ckpt file
    model.load_state_dict(torch.load(model_saved_name, map_location=device))
    


    magData = Maglib.MagLoader()
    magData.b=np.array(B_waveform)
    magData.freq=np.array(Freq)
    magData.temp=np.array(Temp)

    # re-sample
    newStep= 128
    b_buff=np.zeros([magData.b.shape[0],newStep])
    for i in range(magData.b.shape[0]):
        x= np.linspace(0, newStep, magData.b.shape[1], endpoint=True)
        y= magData.b[i]

        k = newStep/magData.b.shape[1]
        b = np.interp(np.arange(0, newStep), x, y)

        b_buff[i]=b
    magData.b=b_buff

    # standardize ###############################################################
    magData.b=std_b.std(magData.b)
    magData.freq=std_freq.std(magData.freq)
    magData.temp=std_temp.std(magData.temp)


    x_data = np.zeros([magData.b.shape[0], magData.b.shape[1], 3])
    x_data[:, :, 0] = magData.b
    x_data[:, :, 1] = magData.freq
    x_data[:, :, 2] = magData.temp

    idx = 0
    dataNums = magData.freq.shape[0]
    # no more than 2000
    #if(dataNums>6000):dataNums=6000

    with torch.no_grad():
        x_data = x_data[idx:idx + dataNums, :, :]
        # Now we can pass a batch of sequences through the model
        inputs = torch.tensor(x_data, dtype=torch.float32)
        if function_use=="valid":
            outputs = model.valid(inputs)
        else:
            outputs = model(inputs)
        
        outputs = outputs.detach().numpy()
        outputs = std_loss.unstd(outputs)
    return outputs

def MagLoss_percise(
    B_waveform,
    Temp,
    Freq,
    model_saved_name="model.ckpt",
    dataset_path=r"data\std_dataset",
    test_num=100,
    ):

    pred_losses=np.zeros((Freq.shape[0],test_num))
    for i in range(test_num):
        pred_losses[:,i:i+1] = MagLoss(
        B_waveform,
        Temp,
        Freq,
        model_saved_name,
        dataset_path,
        function_use="training",)

    pred_loss_avg=np.mean(pred_losses,axis=1)
    pred_loss_avg=pred_loss_avg[:,np.newaxis]
    return pred_loss_avg



def Mag_plot(material_name,relative_error,save_path="",xlim=None):

    plt.figure(figsize=(6,3),dpi=300)
    # change to timesnewroman font
    plt.rcParams["font.family"] = "Times New Roman"

    relv_err=relative_error
    relv_err = np.clip(relv_err, 0, 200)
    relv_err*=100 # convert to percentage
    plt_perc95=np.percentile(relv_err,95)
    plt_perc99=np.percentile(relv_err,99)
    plt_avg=np.mean(relv_err)
    plt_max=np.max(relv_err)


    subtitle=("Avg="+str(round(plt_avg,2))+"%,")
    subtitle+=(" 95-Prct="+str(round(plt_perc95,2))+"%,")
    subtitle+=(" 99-Prct="+str(round(plt_perc99,2))+"%,")
    subtitle+=(" Max="+str(round(plt_max,2))+"%")


    plt.title("Error distribution for "+material_name+"\n",fontsize=18)
    #plt.title("Error distribution for "+"\n",fontsize=10)

    plt.suptitle("\n"+subtitle,fontsize=10)


    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.xlabel("Relative Error of Core Loss [%]",fontsize=15)
    plt.ylabel("Ratio of Data Points",fontsize=15)

    # draw vertical line at 95% percentile and mean
    # plt.axvline(plt_perc95, color='r', linestyle='dashed', linewidth=1, label="95% percentile")
    # plt.axvline(plt_avg, color='g', linestyle='dashed', linewidth=1, label="mean")
    #plt.legend()

    # Set the font size for the axis ticks
    plt.tick_params(labelsize=15)
    
    # get density val for input x
    def get_density(x,RV):
        hist,edge=np.histogram(relv_err, bins=20, density=True)
        for i in range(len(edge)-1):
            if x>=edge[i] and x<edge[i+1]:
                return hist[i]
        return 0

    

    plt.hist(relv_err, bins=20,edgecolor='black', density=True,linewidth=0.5)

    # Plot vertical lines and text for key statistics
    for y_val, stat_func, label in zip(
        [0.07, 0.04, 0.02], 
        [np.mean, lambda x: np.percentile(x, 95), np.max], 
        ["Avg", "95-Prct", "Max"]
    ):
        stat_val = stat_func(np.abs(relv_err))
        y_val = get_density(stat_val,relv_err)+0.001
        plt.plot([stat_val, stat_val], [0, y_val], '--', color="red", linewidth=1)
        plt.text(stat_val + 0.25, y_val, f'{label}={stat_val:.2f}%', color="red", fontsize=10)

    if xlim!=0:
        plt.xlim(0, xlim)

    if save_path!="":
        plt.savefig(save_path, bbox_inches='tight')
        print("plot saved to "+save_path)

    plt.show()

