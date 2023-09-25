import logging
import shutil
from datasets import load_dataset
import os
from tqdm import tqdm
import argparse













def Download(language):
    # Set up logging
    logging.basicConfig(filename='error.log', level=logging.ERROR)

    # Load the dataset
    #load_dataset("mozilla-foundation/common_voice_13_0", "hi", split="train")
    dataset = load_dataset("mozilla-foundation/common_voice_11_0", language ,split="train")
    #dataset = load_dataset("mozilla-foundation/common_voice_11_0", language)

    num_lang_dirs = 20

    # Specify the output directory to move the audio files to
    output_dir = "./Common_voice_Train/"

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Get the total number of examples in the train split
    num_examples = len(dataset["path"])

    # Loop over the examples in the train split and move the audio files
    for i, (audio_path, transcription) in tqdm(enumerate(zip(dataset["path"], dataset["sentence"])), total=num_examples):
        #print(transcription)
        # Get the path to the audio file
        #audio_path = example["path"]

        # Get the transcription
        #transcription = example["sentence"]
        # Check if the transcription contains any of the specified words
        if "tuito" not in transcription and "tuto" not in transcription and "aide" not in transcription and "secours" not in transcription:
            # Get the language index
            lang_index = i % num_lang_dirs
            # Construct the full path to the audio file
            output_dir_ = os.path.join(output_dir, f"{language}/{lang_index}/")
            if not os.path.isdir(output_dir_):
                os.makedirs(output_dir_)
            full_audio_path = os.path.join(output_dir_, os.path.basename(audio_path))

            
            A = audio_path
            print(A)
            B = A.split('/')
            B.pop()



            #print('/', os.path.join(os.path.join(*B)))
            #print( os.listdir(os.path.join(os.path.join(*B), '/')))

            #for each_direcory in os.listdir(os.path.join(os.path.join(*B), '/')):
            for il in range(1000):
                audio_path = os.path.join('/',*B,language+'_train_'+str(il),A.split('/').pop())
                
                try:
                        # Move the audio file to the output directory
                        shutil.move(audio_path, full_audio_path)

                except:# Exception as e:
                        # Log the error and print the exception
                       pass
                       # logging.error(f"Error moving audio file: {e}")
                       # print(f"Error moving audio file: {e}")




if __name__ == "__main__":
            
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--language", type=str,
                        help="language_to_download")    
    args = parser.parse_args()
    Download(args.language)