import os
import subprocess
import multiprocessing
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

#ncentroids-500-subset_size-100M.csv



# Define the number of GPUs to use for downloading
num_gpus = 4

"""
def download_youtube_audios(df, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get available GPU devices
    available_gpus = list(range(len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))))
    num_gpus = len(available_gpus)

    # Iterate over the rows of the dataframe
    for i, row in df.iterrows():
        # Get video ID and start time from dataframe
        video_id = row[0]
        start_time = row[1]

        # Construct URL for the video with start time
        video_url = f"https://www.youtube.com/watch?v={video_id}&t={start_time}"

        # Construct output file path for the audio
        output_file = os.path.join(output_dir, f"{video_id}_{start_time}.wav")

        # Check if output file already exists
        if os.path.exists(output_file):
            print(f"Skipping {video_id}_{start_time} as it already exists")
            continue

        # Choose a GPU device to use
        gpu_index = i % num_gpus
        gpu_device = available_gpus[gpu_index]
        print(f"Using GPU device {gpu_device} for {video_id}_{start_time}")

        # Construct FFmpeg command to extract audio and save to file
        cmd = f"ffmpeg -ss {start_time} -i $(youtube-dl -f 'bestaudio[ext=m4a]' --get-url '{video_url}') -t 10 -acodec pcm_s16le -ac 1 -ar 16000 -f wav - | sox - -r 16000 {output_file}"
        
        # Run FFmpeg command using subprocess with the chosen GPU device
        subprocess.run(["CUDA_VISIBLE_DEVICES", str(gpu_device), "bash", "-c", cmd])


"""
import os
import youtube_dl
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd



def extract_audio(video_id, start_time, output_file):
    # Construct URL for the video with start time
    video_url = f"https://www.youtube.com/watch?v={video_id}&t={start_time}"
    
    # Define options for youtube_dl
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]',
        'quiet': True,
    }

    # Download the audio with youtube_dl and save to file
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        url = ydl.extract_info(video_url, download=False)['url']
        cmd = f"youtube-dl -f 'bestaudio[ext=m4a]' -o - '{video_url}' | ffmpeg -ss {start_time} -i pipe:0 -to 10 -acodec pcm_s16le -ac 1 -ar 16000 -f wav - | sox -r 16000 -t wav - {output_file}"

        os.system(cmd)



def download_youtube_audios(df, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over the rows of the dataframe
    tasks = []
    for i, row in tqdm(df[:20000000].iterrows()):
        # Get video ID and start time from dataframe
        video_id = row[0]
        start_time = row[1]

        # Construct output file path for the audio
        output_file = os.path.join(output_dir, f"{video_id}_{start_time}.wav")

        # Check if output file already exists
        if os.path.exists(output_file):
            print(f"Skipping {video_id}_{start_time} as it already exists")
            continue

        # Add task to list
        tasks.append((video_id, start_time, output_file))

    # Create a pool of processes to run the tasks
    num_processes = cpu_count()*2
    pool = Pool(processes=num_processes)

    # Use the pool to run the tasks in parallel
    results = []
    with tqdm(total=len(tasks), desc="Downloading videos") as pbar:
        for t in tasks:
            result = pool.apply_async(extract_audio, args=(t[0], t[1], t[2]))
            results.append(result)
            pbar.update(1)

    # Wait for all tasks to complete
    pool.close()
    pool.join()

    # Print results
    for result in results:
        result.get()
        
    print("All videos downloaded successfully!")


df = pd.read_csv('ncentroids-500-subset_size-100M.csv', header=None, names=['id', 'start_time'])

download_youtube_audios(df, 'Video_Audio')
