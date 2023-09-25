# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:27:12 2023

@author: hafsa
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:09:53 2023

@author: hafsa
"""

import os
import collections
import numpy as np
from numpy.lib.format import open_memmap
from pathlib import Path
from tqdm import tqdm
import openwakeword
import openwakeword.data
import openwakeword.utils
import openwakeword.metrics
import random
import scipy
import datasets
import matplotlib.pyplot as plt
import torch
from torch import nn
#import IPython.display as ipd
import torchaudio 
import random
import soundfile as sf
from tqdm import tqdm
import os 
import librosa
import uuid
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import argparse
import warnings

warnings.filterwarnings('ignore')

import os
import numpy as np
from tqdm import tqdm
import concurrent.futures
import time
import numpy as np
import torch

import torch.nn as nn





if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    negative_features = np.load("./New_Features/Negative_newmp3_resampled_1.npy")
    #negative_featuresI = np.load("Negative_Libri.npy")

    #
    Acc_before = 0    
    parser = argparse.ArgumentParser()
    parser.add_argument("-W","--Word", type=str,help="True to compute negative examples and False for positive examples")

    args = parser.parse_args()
    Word  = args.Word
    if Word == 'tuito':
        positive_features  = np.load("New_Features/Positive_newtuito.npy")
        output_path = "./Tuito_1s_Speaker_1.onnx"
        figname = 'tuto_recal.png'
        #negative_features = np.concatenate((negative_features, negative_featuresI), axis=0)
    elif Word == 'aide':
        positive_features  = np.load("New_Features/Positive_newaide.npy")
        #negative_features = np.concatenate((negative_features, negative_featuresI), axis=0)

        output_path = "./Aide_1s_Speaker_1.onnx"
        figname = 'aide_recal.png'
    elif Word == 'secours':
        positive_features  = np.load("New_Features/Positive_newsecours.npy")
        #negative_features = np.concatenate((negative_features, negative_featuresI ), axis=0)

        output_path = "./Secours_1s_Speaker_1.onnx"
        figname = 'secours_recal.png'
    elif Word == 'help':
        positive_features  = np.load("New_Features/Positive_help.npy")
        #negative_features = np.concatenate((negative_features, negative_featuresI), axis=0)

        output_path = "./Help.onnx"
        figname = 'help_recal.png'
    

    NumberOfSamples = positive_features.shape[0]

    list_OfFeatures = os.listdir('./New_Features')
    print(list_OfFeatures)
    list_Negative = [file for file in list_OfFeatures if 'Negative' in file]
    print(list_Negative)    

    NumberOfexample = int(positive_features.shape[0]/ len(list_Negative))

    
    """
    X =  positive_features #np.vstack((positive_features[:,:] ,positive_features_secours[:,:], positive_features_aide[:,:]  ))

    len_negative = 0
    for negative_features_file in tqdm(list_Negative[:]):
        negative_features = np.load(os.path.join("New_Features",negative_features_file))
        X = np.vstack((X, negative_features))
        len_negative = len_negative +  len(negative_features)



    y = np.array([1]*len(positive_features) + [0]*len_negative).astype(np.float32)[...,None]
    """

    def load_negative_feature_file(file_path):
        try:
            with open(file_path, 'rb') as f:
                return np.load(f)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return None

    # Get the list of negative feature files
    #list_OfFeatures = os.listdir('.')
    #list_Negative = [file for file in list_OfFeatures if 'Negative' in file and 'ESC' not in file and 'Video' not in file]

    # Calculate the number of negative examples to load per positive example
    #NumberOfexample = int(positive_features.shape[0]/ len(list_Negative))

    # Load the positive examples
    X = positive_features
    """    
    # Use a ThreadPoolExecutor to load the negative examples in parallel with a 10-minute timeout
    len_negative = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(load_negative_feature_file, os.path.join(".", file)): file for file in list_Negative[:]}
        for i, future in enumerate(tqdm(concurrent.futures.as_completed(future_to_file), total=len(future_to_file))):
            file = future_to_file[future]
            negative_features = future.result(timeout=300)  # 10 minute timeout
            if negative_features is not None:
                X = np.vstack((X, negative_features))
                len_negative += len(negative_features)
            print(f"Loaded {i+1} of {len(future_to_file)} negative feature files")

    # Create the target array y
    y = np.array([1]*len(positive_features) + [0]*len_negative).astype(np.float32)[...,None]

    """





# Make Pytorch dataloader
  

    # Make Pytorch dataloader

    
    
    save_file = Word+'_1s_Speaker_1.pth'
    
    
    layer_dim = 64
    fcn = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(X.shape[1]*X.shape[2], layer_dim), # since the input is flattened, it's timesteps*feature columns
                        nn.LayerNorm(layer_dim),
                        nn.ReLU(),
                        nn.Linear(layer_dim, layer_dim),
                        nn.LayerNorm(layer_dim),
                        nn.ReLU(),
                        nn.Linear(layer_dim, 1),
                        nn.Sigmoid(),
                    ).to(device)


    
    #fcn = torch.load(save_file)

    """
    if torch.cuda.is_available():
        device_ids = list(range(torch.cuda.device_count()))
        fcn = nn.DataParallel(fcn, device_ids=device_ids)
    fcn = fcn.to('cuda')
    """

    training_data = None    
    y = None
    loss_function = torch.nn.functional.binary_cross_entropy
    optimizer = torch.optim.Adam(fcn.parameters(), lr=0.001)
    

    
    n_epochs = 20
    history = collections.defaultdict(list)




    for i in tqdm(range(n_epochs), total=n_epochs):
        np.random.shuffle(positive_features)   
        random.shuffle(list_Negative)
        tp = 0 
        fn = 1
        fcn.to(device)
        cpt_acc = 0
        Acc_TP =  0
        Acc_TN = 0    
        X = positive_features
        cpt_load = 0
    # Use a ThreadPoolExecutor to load the negative examples in parallel with a 10-minute timeout
        
        for j in range(4):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                del X 
                del training_data
                del y
                len_negative = 0
                #if j==0:
                    #X = positive_features
                #else:
                X = positive_features[j*int(positive_features.shape[0]/4):j*int(positive_features.shape[0]/4)+ int(positive_features.shape[0]/4) ,:,:]
                len_positive_todeclare = X.shape[0]




                future_to_file = {executor.submit(load_negative_feature_file, os.path.join("New_Features", file)): file for file in list_Negative[cpt_load:cpt_load+70]}
                
                for i, future in enumerate(tqdm(concurrent.futures.as_completed(future_to_file), total=len(future_to_file))):
                    file = future_to_file[future]
                    negative_features = future.result(timeout=300)  # 10 minute timeout
                    if negative_features is not None: 
                        X = np.vstack((X, negative_features))
                        len_negative += len(negative_features)
                        
                    #print(f"Loaded {i+1} of {len(future_to_file)} negative feature files")

            # Create the target array y
            #if j==0:
            y = np.array([1]*len_positive_todeclare + [0]*len_negative).astype(np.float32)[...,None]
           




            batch_size = 1024
            training_data = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
            batch_size = batch_size,
            shuffle = True
            )

            for batch in training_data:
                x, y = batch[0].to(device), batch[1].to(device)
                cpt_acc += 1
                
                # Get weights for classes, and assign 10x higher weight to negative class
                # to help the model learn to not have too many false-positives
                # As you have more data (both positive and negative), this is less important
                
                weights = 1*torch.ones(y.shape[0])
                weights[y.flatten() == 1] = 1
                weights[y.flatten() == 0] = 1    
                # Zero gradients
                
                optimizer.zero_grad()
                # Run forward pass
                
                predictions = fcn(x)
                # Update model parameters
                
                loss = loss_function(predictions, y, weights[..., None].to(device))
                #print(loss)
                
                loss.backward()
                optimizer.step()
                
                ## accuracy 

                Acc_TP  =  Acc_TP + np.array(sum(predictions.flatten().detach().cpu().numpy()[y.flatten().detach().cpu().numpy() == 1] >= 0.5)/( np.count_nonzero(y.flatten().detach().cpu().numpy() ==  1)+1e-6))
                Acc_TN  =  Acc_TN+ np.array(sum(predictions.flatten().detach().cpu().numpy()[y.flatten().detach().cpu().numpy() == 0] <= 0.5) /( np.count_nonzero(y.flatten().detach().cpu().numpy() ==  0)+1e-6))    



                # Log metrics
                history['loss'].append(float(loss.detach().cpu().numpy()))

                if y.eq(1).any():
                    tp = np.array(sum(predictions.flatten().detach().cpu().numpy()[y.flatten().detach().cpu().numpy() == 1] >= 0.5))
                    fn = np.array(sum(predictions.flatten().detach().cpu().numpy()[y.flatten().detach().cpu().numpy() == 1] < 0.5))
                
                history['recall'].append(float(tp/(tp+fn)))
    



        if (Acc_TP/cpt_acc+Acc_TN/cpt_acc)/2 > Acc_before:
            print(f'Saved AC_now:{Acc_before}')
            print(f'Saved AC_TP:{Acc_TP/cpt_acc}')
            print(f'Saved AC_TN:{Acc_TN/cpt_acc}')
            torch.save(fcn, save_file)
            Acc_before = (Acc_TP/cpt_acc+Acc_TN/cpt_acc)/2
            torch.onnx.export(fcn.to('cpu'), args=torch.zeros((1, 3, 96)).to('cpu'), f=output_path) # the 'args' is the shape of a single example*

            plt.figure(f'AC_TN:{Acc_TN/cpt_acc}')
            plt.plot(history['loss'], label="loss")
            plt.plot(history['recall'], label="recall")
            plt.legend()
            plt.ylim(0,1.5)
            plt.savefig(figname)
        else:
            print(f'Could not Save AC_TP:{Acc_TP/cpt_acc}')
            print(f'Could not Save AC_TN:{Acc_TN/cpt_acc}')


    

    
    torch.onnx.export(fcn.to('cpu'), args=torch.zeros((1, 3, 96)).to('cpu'), f=output_path) # the 'args' is the shape of a single example

    plt.figure()
    plt.plot(history['loss'], label="loss")
    plt.plot(history['recall'], label="recall")
    plt.legend()
    plt.ylim(0,1.5)
    plt.savefig(figname)
    
    
    





