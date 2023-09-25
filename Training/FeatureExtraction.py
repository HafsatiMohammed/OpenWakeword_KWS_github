# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:06:54 2023

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

import argparse
import warnings
import numpy as np
import wave 



warnings.filterwarnings('ignore')


def float2pcm(sig, dtype='int16'):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)








def olafilt(b, x, zi=None):
    """
    Filter a one-dimensional array with an FIR filter
    Filter a data sequence, `x`, using a FIR filter given in `b`.
    Filtering uses the overlap-add method converting both `x` and `b`
    into frequency domain first.  The FFT size is determined as the
    next higher power of 2 of twice the length of `b`.
    Parameters
    ----------
    b : one-dimensional numpy array
        The impulse response of the filter
    x : one-dimensional numpy array
        Signal to be filtered
    zi : one-dimensional numpy array, optional
        Initial condition of the filter, but in reality just the
        runout of the previous computation.  If `zi` is None or not
        given, then zero initial state is assumed.
    Returns
    -------
    y : array
        The output of the digital filter.
    zf : array, optional
        If `zi` is None, this is not returned, otherwise, `zf` holds the
        final filter delay values.
    """

    L_I = b.shape[0]
    # Find power of 2 larger that 2*L_I (from abarnert on Stackoverflow)
    L_F = 2<<(L_I-1).bit_length()
    L_S = L_F - L_I + 1
    L_sig = x.shape[0]
    offsets = range(0, L_sig, L_S)

    # handle complex or real input
    if np.iscomplexobj(b) or np.iscomplexobj(x):
        fft_func = np.fft.fft
        ifft_func = np.fft.ifft
        res = np.zeros(L_sig+L_F, dtype=np.complex128)
    else:
        fft_func = np.fft.rfft
        ifft_func = np.fft.irfft
        res = np.zeros(L_sig+L_F)

    FDir = fft_func(b, n=L_F)

    # overlap and add
    for n in offsets:
        res[n:n+L_F] += ifft_func(fft_func(x[n:n+L_S], n=L_F)*FDir)

    if zi is not None:
        res[:zi.shape[0]] = res[:zi.shape[0]] + zi
        return res[:L_sig], res[L_sig:]
    else:
        return res[:L_sig]
    

def Add_noise(s,n,SNR):
    #SNR = 10**(SNR/20)
    Es = np.sqrt(np.sum(s[:]**2)+1e-6)
    En = np.sqrt(np.sum(n[:]**2)+1e-6)
    iSNR = 10*np.log10(Es**2/(En**2+1e-6)) 
    alpha = 10**((iSNR-SNR)/20)
    
    #alpha = Es/(SNR*(En+1e-8))
    #•Mix = s+alpha*n[0:160000]
    
    return  alpha
"""
def Download_noise(noise_dir,ListOfNoises, K):
    noiseToAdd = random.choices(ListOfNoises, k = int((K)/1))
    Noise = []
    for each_noise in noiseToAdd:
        #print(each_noise)
        Noisei, sr = librosa.load(os.path.join(noise_dir , each_noise), sr=16000)
        #audio_object_reader = wave.open(os.path.join(noise_dir , each_noise), 'rb')                   
        #Noisei = audio_object_reader.readframes(audio_object_reader.getnframes()) 
        #audio_object_reader.close()



        #print(Noisei)
        if np.isnan(Noisei).any() == False:
            if len(Noisei) < (16000*1.5):
                #print(hello)
                Noisei = np.concatenate( (Noisei, np.zeros((16000-len(Noise) )) ))
            for iSeq in range(int(len(Noisei)/16000)):
                Noise.append(Noisei[iSeq*16000: iSeq*16000+16000 ])
        else:
            print('Nan from Silence')

    return Noise
"""

def Preprocessing_Mix(S, N, SRIR, Desired_SIR, samples):
    
    if len(S)<samples:
        S = np.concatenate( (S, np.zeros((16000-len(S) )) )) 
    else:
        S = S[:samples]
    Contrib_Rev = olafilt(SRIR[0][0,:,0], S)[:samples]
    Contrib_N = olafilt(SRIR[1][0,:,0], N)[:samples]
    alpha = Add_noise(Contrib_Rev,Contrib_N,Desired_SIR)
    Contrib_N = alpha*Contrib_N
    Mix = Contrib_Rev + Contrib_N
    
    return Mix



def Load_SRIR():  
    SRIR_Dir = r'../Speech_Enhancement/RI_Train'
    SRIR = np.zeros((6,5,18,32682,3))
    #print(os.listdir(SRIR_Dir))
    for eachfile in tqdm(os.listdir(SRIR_Dir)):   
        Name = eachfile.split('_')
        file = os.path.join(SRIR_Dir, eachfile)
        matlabfile = scipy.io.loadmat(file)
        ImpResp_Rev = matlabfile['ImpResp_Rev'];
        ImpResp_Rev = ImpResp_Rev[:,0]
        secs = len(ImpResp_Rev[0])/48000 # Number of seconds in signal X
        samps = secs*16000    # Number of samples to downsample
        ImpResp_Rev_ = np.zeros((32682, 3))
        ImpResp_Rev_[:,0] = np.squeeze(scipy.signal.resample(ImpResp_Rev[0],int(samps))[:], 1)
        ImpResp_Rev_[:,1] = np.squeeze(scipy.signal.resample(ImpResp_Rev[1],int(samps))[:],1)
        ImpResp_Rev_[:,2] = np.squeeze(scipy.signal.resample(ImpResp_Rev[2],int(samps))[:],1)
        SRIR[int(Name[1])-1, int(Name[2])-1 , int(Name[3].split('.')[0])-1,:,: ] = ImpResp_Rev_# scipy.signal.resample(ImpResp_Rev_[:,0], int(samps))
    return SRIR




def Preprocessing(S, SRIR):
    
    if len(S)<16000*2:
        S = np.concatenate( (S, np.zeros((16000-len(S) )) )) 
    else:
        S = S[:16000*2]
    Contrib_Rev = olafilt(SRIR[0][0,:,0], S)[:16000*2]
    
    Mix = Contrib_Rev
    
    #randomNoiseFactor = random.uniform(0.05, 0.2)
    #Mix = addNoise(Contrib_Rev, Contrib_N, randomNoiseFactor)
    return Mix#/np.max(np.abs(Mix)+1e-6)

# Get audio embeddings (features) for negative clips and save to .npy file
# Process files by batch and save to Numpy memory mapped file so that
# an array larger than the available system memory can be created









def compute_feature(Negative, Word, device):

    os.environ["CUDA_VISIBLE_DEVICES"]=device

    F = openwakeword.utils.AudioFeatures()
    
    
    negative_clips, negative_durations = openwakeword.data.filter_audio_paths(
        [Word],
        min_length_secs = 1.0, # minimum clip length in seconds
        max_length_secs = 60*30, # maximum clip length in seconds
        duration_method = "header" # use the file header to calculate duration
    )
    print(f"{len(negative_clips)} negative clips after filtering, representing ~{sum(negative_durations)//3600} hours")
    
    # Use HuggingFace datasets to load files from disk by batches
    audio_dataset = datasets.Dataset.from_dict({"audio": negative_clips})
    audio_dataset = audio_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    
    model = torchaudio.models.ConvTasNet(num_sources=1, enc_kernel_size = 256, enc_num_feats = 256, msk_kernel_size = 3, msk_num_feats= 128, msk_num_hidden_feats = 256, msk_num_layers = 8, msk_num_stacks = 2)
    model.cuda()
    checkpoint = torch.load('../Speech_Enhancement/BestModel_SE_SDR.pth.tar', map_location='cuda')
    model.load_state_dict(checkpoint)
    SRIR =  Load_SRIR()
    if Negative == 1 :
        print(f'Negative = {Negative}')
        batch_size = 10 # number of files to load, compute features, and write to mmap at a time
        clip_size = 1.5  # the desired window size (in seconds) for the trained openWakeWord model
        N_total = int(sum(negative_durations)//clip_size) # maximum number of rows in mmap file
        n_feature_cols = F.get_embedding_shape(clip_size)
        noise_dir=  r'../Speech_Enhancement/Noise_Background'
        
        output_file = "Negative_"+ '_'.join(Word.split('/')[-2:]) + ".npy"
        output_array_shape = (N_total, n_feature_cols[0], n_feature_cols[1])
        fp = open_memmap(output_file, mode='w+', dtype=np.float32, shape=output_array_shape)
        
        row_counter = 0
        for i in tqdm(np.arange(0, audio_dataset.num_rows, batch_size)):
            # Load data in batches and shape into rectangular array
            wav_data = [(j["array"]*32767).astype(np.int16) for j in audio_dataset[i:i+batch_size]["audio"]]
            wav_data = openwakeword.data.stack_clips(wav_data, clip_size=int(16000*clip_size))
            ListOfNoises =os.listdir(noise_dir)
        
            #Noises = Download_noise(noise_dir,ListOfNoises, 10)
            """
            for il in range(wav_data.shape[0]):
                Srir = []
                random_value = random.randint(0, 4)
                Srir.append( SRIR[random_value, random.randint(0, 4) , random.choices(range(18),k=1),:,:])
                Srir.append( SRIR[random_value, random.randint(0, 4) , random.choices(range(18),k=1),:,:])
                #wav_data[il,:] = Preprocessing_Mix(wav_data[il,:], np.reshape(np.array(Noises), np.zeros(wav_data[il,:].shape), Srir, random.randint(0, 100), int(16000*1.5))
            """
            wav_data = torch.from_numpy(np.expand_dims(wav_data,1))
          
            #with torch.no_grad():
              #wav_data = model(wav_data.cuda().float())
              
            wav_data = float2pcm(wav_data[:,0,:].cpu().detach().numpy()/(np.repeat(np.expand_dims(np.max(np.abs(wav_data[:,0,:].cpu().detach().numpy()),1),1), 16000, axis=1) +1e-6))
        
            #wav_data = float2pcm(SE_MODULE.process(wav_data)[0][:,0,:])
            # Compute features (increase ncpu argument for faster processing)
            #plt.plot(wav_data[0,:])
            #plt.show()
            # here add SE
        
            features = F.embed_clips(x=wav_data, batch_size=1024, ncpu=8)
            
            # Save computed features to mmap array file (stopping once the desired size is reached)
            if row_counter + features.shape[0] > N_total:
                fp[row_counter:min(row_counter+features.shape[0], N_total), :, :] = features[0:N_total - row_counter, :, :]
                fp.flush()
                break
            else:
                fp[row_counter:row_counter+features.shape[0], :, :] = features
                row_counter += features.shape[0]
                fp.flush()
                
        # Trip empty rows from the mmapped array
        openwakeword.data.trim_mmap(output_file)
    else:
        # Get positive example paths, filtering out clips that are too long or too short

        
        Data_Positivei = os.path.join("TTS_TUITO/TrainingSamples/", Word)
        Data_Positive = os.path.join("TTS_TUITO/", Word)
        
        for cpt, each_file in tqdm(enumerate(os.listdir(Data_Positive))):
            if cpt>-1:
              wav_file = os.path.join(Data_Positive,each_file)
              data , sr = librosa.load(wav_file, sr=16000)
              #audio_object_reader = wave.open(wav_file, 'rb')                    
              #data = audio_object_reader.readframes(audio_object_reader.getnframes()) 
              #audio_object_reader.close()

              if len(data)>int(16000*2):
                data = data[:int(16000*2)]
              #print(data.shape)
              wav_name =  os.path.join(Data_Positivei,str(uuid.uuid1())+'.wav')

              sf.write(wav_name, data, 16000)
              for jl in range(6):
                  Srir = []
                  random_value = random.randint(0, 4)
                  Srir.append( SRIR[random_value, random.randint(0, 4) , random.choices(range(18),k=1),:,:])
                  wav_name =  os.path.join(Data_Positivei,str(uuid.uuid1())+'.wav')
                  
                  data_rev = Preprocessing(data, Srir)
                  if jl==0:
                      data_rev  = data
                  sf.write(wav_name, data_rev, 16000)
        
                
        
        negative_clips, negative_durations = openwakeword.data.filter_audio_paths(
        '/home/KWS_EfficientNet/Youtube/mp3/'+random.randint(0,20),
        min_length_secs = 1.0, # minimum clip length in seconds
        max_length_secs = 60*30, # maximum clip length in seconds
        duration_method = "header" # use the file header to calculate duration
        )
        print(f"{len(negative_clips)} negative clips after filtering, representing ~{sum(negative_durations)//3600} hours")

        
        positive_clips, durations = openwakeword.data.filter_audio_paths(
            [
                Data_Positivei
            ],
            min_length_secs = 0, # minimum clip length in seconds
            max_length_secs = 2, # maximum clip length in seconds
            duration_method = "header" # use the file header to calculate duration
        )
        
        
        #print()
        #print(negative_clips.shape)
        
        print(f"{len(positive_clips)} positive clips after filtering")
        #print(durations)
        sr = 16000
        total_length_seconds = 1.5 # must be the same window length as that used for the negative examples
        total_length = int(16000*total_length_seconds)
        
        """
        jitters = (np.random.uniform(0, 0.2, len(positive_clips))*sr).astype(np.int32)
        starts = [total_length - (int(np.ceil(i*sr))+j) for i,j in zip(durations, jitters)]
        starts = [0 for j in starts]
        print(starts)
        ends = [16000 for j in  starts]
        print(ends)
        """

        jitters = (np.random.uniform(0, 0.2, len(positive_clips))*sr).astype(np.int32)
        starts = [total_length - (int(np.ceil(i*sr))+j) for i,j in zip(durations, jitters)]
        ends = [int(i*sr) + j for i, j in zip(durations, starts)]


        # Create generator to mix the positive audio with background audio
        batch_size = 8
        mixing_generator = openwakeword.data.mix_clips_batch(
            foreground_clips = positive_clips,
            background_clips = negative_clips,
            combined_size = total_length,
            batch_size = batch_size,
            snr_low = 5,
            snr_high = 20,
            start_index = starts,
            volume_augmentation=True, # randomly scale the volume of the audio after mixing
        )
    
    
        N_total = len(positive_clips) # maximum number of rows in mmap file
        n_feature_cols = F.get_embedding_shape(total_length_seconds)
        
        output_file = "Positive_"+ Word+ ".npy"
        
        output_array_shape = (N_total*10, n_feature_cols[0], n_feature_cols[1])
        
        fp = open_memmap(output_file, mode='w+', dtype=np.float32, shape=output_array_shape)
        
        row_counter = 0
        for batch in tqdm(mixing_generator, total=N_total//batch_size):
            batch, lbls, background = batch[0], batch[1], batch[2]
            # Compute audio features
            #batch = np.expand_dims(batch,1)
        
            for numberOfRandomness in range (1):
                  for il in range(batch.shape[0]):
                      Srir = []
                      random_value = random.randint(0, 1)
                      #batch[il,:] = Preprocessing(batch[il,:], Srir)    
                  batch = torch.from_numpy(np.expand_dims(batch,1))
                  #if random.randint(0, 10)>5:
                  with torch.no_grad():
                      batch = model(batch.cuda().float())
                  #else:
                      #batch = batch.cuda().float()
                  batch = float2pcm(batch[:,0,:].cpu().detach().numpy()/(np.repeat(np.expand_dims(np.max(np.abs(batch[:,0,:].cpu().detach().numpy()),1),1), batch.shape[2], axis=1)+1e-6 )) # batch.shape[2] or batch.shape[0] samples
                  features = F.embed_clips(batch, batch_size=256)
                  # Save computed features
                  fp[row_counter:row_counter+features.shape[0], :, :] = features
                  row_counter += features.shape[0]
                  fp.flush()
                  
                  if row_counter >= N_total:
                      break
        
        # Trip empty rows from the mmapped array
        openwakeword.data.trim_mmap(output_file)       
        
        
if __name__ == "__main__":
            
    parser = argparse.ArgumentParser()
    parser.add_argument("-Neg","--Negative", type=int,
                        help="True to compute negative examples and False for positive examples")
    parser.add_argument("-W", "--Word", type=str,
                        help="In case negative is false you should give the name of the positive word")
    parser.add_argument("-D", "--Device", type=str,
                        help="Cuda Device")    
    args = parser.parse_args()
    compute_feature(args.Negative, args.Word, args.Device)
    



    
    

    



