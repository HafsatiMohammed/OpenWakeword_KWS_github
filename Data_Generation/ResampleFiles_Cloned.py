import os
import subprocess
from multiprocessing import Pool

input_folder = "Cloned_voices/tuito/"
output_folder = "Cloned_voices/tuito_16K/"

# create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def convert_file(filename):
    # construct input and output file paths
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "_16K.wav")

    # check if file is a WAV file
    if filename.endswith(".wav"):
        # run FFmpeg command to resample audio to 16KHz
        try:
            cmd = ["ffmpeg", "-i", input_path, "-ar", "16000", output_path]
            subprocess.call(cmd)
            print(f"{filename} converted successfully")
        except:
            print('problem')
    else:
        print(f"{filename} is not a WAV file, skipping")

if __name__ == '__main__':
    # get list of files in input folder

  input_folder = "Cloned_voices/tuito/"
  output_folder = "Cloned_voices/tuito_16K/"
   
# create output folder if it doesn't exist
  if not os.path.exists(output_folder):
        os.makedirs(output_folder)

  files = os.listdir(input_folder)

    # create a process pool with 4 workers
  with Pool(processes=10) as pool:
        # run the convert_file function for each file in parallel
        pool.map(convert_file, files)


  input_folder = "Cloned_voices/secours/"
  output_folder = "Cloned_voices/secours_16K/"

# create output folder if it doesn't exist
  if not os.path.exists(output_folder):
        os.makedirs(output_folder)

  files = os.listdir(input_folder)

    # create a process pool with 4 workers
  with Pool(processes=10) as pool:
        # run the convert_file function for each file in parallel
        pool.map(convert_file, files)




  input_folder = "Cloned_voices/aide/"
  output_folder = "Cloned_voices/aide_16K/"

# create output folder if it doesn't exist
  
  if not os.path.exists(output_folder):
        os.makedirs(output_folder)

  files = os.listdir(input_folder)

    # create a process pool with 4 workers
  with Pool(processes=10) as pool:
        # run the convert_file function for each file in parallel
        pool.map(convert_file, files)


