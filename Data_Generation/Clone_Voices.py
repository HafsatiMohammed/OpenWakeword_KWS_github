from TTS.api import TTS
import os 
import random 
import uuid
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"







Output_dir = 'Cloned_voices'
Source_dir = '/home/mhafsati/KWS_EfficientNet/TTS_TUITO_New'
Target_dir = '/home/mhafsati/KWS_EfficientNet/Common_voice_Train_resampled/fr/0'

directories = [ 'aide_real','secours_forClone', 'tuito_forClone']

numberOfGeneration = 100000-7000
tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False, gpu=True)


for directory in directories:
	Source_directory = os.path.join(Source_dir, directory)
	Output_directory = os.path.join(Output_dir, directory)
	ListofTagets = os.listdir(Target_dir)
	ListofSources = os.listdir(Source_directory)
	if not os.path.exists(Output_directory):
		os.makedirs(Output_directory)


	for i in tqdm(range(numberOfGeneration)):
		random_target = random.choice(ListofTagets)
		random_source = random.choice(ListofSources)
		Source_File = os.path.join(Source_directory,random_source)
		Target_File = os.path.join(Target_dir,random_target)
		Output_File = os.path.join(Output_directory,str(uuid.uuid4()))+'.wav'

		tts.voice_conversion_to_file(source_wav=Source_File, target_wav=Target_File, file_path=Output_File)



















