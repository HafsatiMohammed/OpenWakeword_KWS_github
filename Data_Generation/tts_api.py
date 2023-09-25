import itertools
import os 
from google.cloud import texttospeech
from pydub import AudioSegment
import json
import uuid
import numpy as np
import random
import sys
import time
from tqdm import tqdm
import argparse



def main(args):
    pathwav = 'C:\\Users\\Admin\\Documents\\Dev\\madmin-server\\public\\voices'
    # Credentials 
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=  'golden-rite-386813-80a098d3d2f5.json'    #'charged-magnet-377214-f74c9844ac76.json'
    # Instantiates a client
    client = texttospeech.TextToSpeechClient()
    # Generate TTS Audio Files 
    def tts_audio_generationfemale(text, language_code, name, pitch, audio_path, speed):
        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=text)
        # Build the voice request, select the language code ("en-US") and the ssml
        # voice gender ("neutral")    
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code, 
            name=name,
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        # Select the type of audio file you want returned
        audio_config = texttospeech.AudioConfig(
            pitch=pitch,
            speaking_rate=speed,
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000 )
        # Perform the text-to-speech request on the text input with the selected
        # voice parameters and audio file type
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        # The response's audio_content is binary.
        with open(audio_path, "wb") as out:
            # Write the response to the output file.
            out.write(response.audio_content)
        # Load the audio file and crop or pad to 2 seconds
        sound = AudioSegment.from_file(audio_path)
        if len(sound) > 2000:
            sound = sound[:2000]
        else:
            sound = sound + AudioSegment.silent(duration=2000-len(sound))
        # Set the frame rate to 16K
        sound.set_frame_rate(16000)
        # Export the audio file as WAV format
        sound.export(audio_path, format="wav")
    def tts_audio_generationmale(text, language_code, name, pitch, audio_path, speed):
        # Set the text input to be synthesized
        synthesis_input = texttospeech.SynthesisInput(text=text)
        # Build the voice request, select the language code ("en-US") and the ssml
        # voice gender ("neutral")    
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code, 
            name=name,
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )
        # Select the type of audio file you want returned
        audio_config = texttospeech.AudioConfig(
            pitch=pitch,
            speaking_rate=speed,
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000
        )
        # Perform the text-to-speech request on the text input with the selected
        # voice parameters and audio file type
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        # The response's audio_content is binary.
        with open(audio_path, "wb") as out:
            # Write the response to the output file.
            out.write(response.audio_content)
        # Load the audio file and crop or pad to 2 seconds
        sound = AudioSegment.from_file(audio_path)
        if len(sound) > 2000:
            sound = sound[:2000]
        else:
            sound = sound + AudioSegment.silent(duration=2000-len(sound))
        # Set the frame rate to 16K
        sound.set_frame_rate(16000)
        # Export the audio file as WAV format
        sound.export(audio_path, format="wav")

    config = [
        {'speaker':'SP003',"name":"fr-FR-Wavenet-A", "pitch":0, "gender":"Female", "db": 0.0},
        {'speaker':'SP003',"name":"fr-FR-Wavenet-B", "pitch":0, "gender":"Male", "db": 0.0},
        {'speaker':'SP003',"name":"fr-FR-Wavenet-C", "pitch":0, "gender":"Female", "db": 0.0},
        {'speaker':'SP003',"name":"fr-FR-Wavenet-D", "pitch":0, "gender":"Male", "db": 0.0},
        {'speaker':'SP003',"name":"fr-FR-Wavenet-E", "pitch":0, "gender":"Female", "db": 0.0},


        {'speaker':'SP003',"name":"fr-FR-Neural2-A", "pitch":0, "gender":"Female", "db": 0.0},
        {'speaker':'SP003',"name":"fr-FR-Neural2-B", "pitch":0, "gender":"Male", "db": 0.0},
        {'speaker':'SP003',"name":"fr-FR-Neural2-C", "pitch":0, "gender":"Female", "db": 0.0},
        {'speaker':'SP003',"name":"fr-FR-Neural2-D", "pitch":0, "gender":"Male", "db": 0.0},
        {'speaker':'SP003',"name":"fr-FR-Neural2-E", "pitch":0, "gender":"Female", "db": 0.0},



        {'speaker':'SP003',"name":"fr-FR-Standard-A", "pitch":0, "gender":"Female", "db": 0.0},
        {'speaker':'SP003',"name":"fr-FR-Standard-B", "pitch":0, "gender":"Male", "db": 0.0},
        {'speaker':'SP003',"name":"fr-FR-Standard-C", "pitch":0, "gender":"Female", "db": 0.0},
        {'speaker':'SP003',"name":"fr-FR-Standard-D", "pitch":0, "gender":"Male", "db": 0.0},
        {'speaker':'SP003',"name":"fr-FR-Standard-E", "pitch":0, "gender":"Female", "db": 0.0},    
        

        {'speaker':'SP003',"name":"fr-CA-Neural2-A", "pitch":0, "gender":"Female", "db": 0.0},
        {'speaker':'SP003',"name":"fr-CA-Neural2-B", "pitch":0, "gender":"male", "db": 0.0},
        {'speaker':'SP003',"name":"fr-CA-Standard-A", "pitch":0, "gender":"female", "db": 0.0},
        {'speaker':'SP003',"name":"fr-CA-Standard-B", "pitch":0, "gender":"male", "db": 0.0},
        {'speaker':'SP003',"name":"fr-CA-Standard-C", "pitch":0, "gender":"female", "db": 0.0},
        {'speaker':'SP003',"name":"fr-CA-Standard-D", "pitch":0, "gender":"male", "db": 0.0},

    ]






    config_l = config[args.configuration]
    Sentenses = ['à l''aide', 'au secours', 'hey tuito']
    FileName =  ['aide_forClone', 'secours_forClone', 'tuito_forClone']

    """    
    Sentenses = [
"Hey plutot",
"hey twiter",
"Hey pepito",
"hey toto",
"Hey Tuivo",
"Hey Tuigo",
"Hey Tuiko",
"Hey Tuita",
"Hey Tuitsu",
"Hey Tuise",
"Au recourt",
"Au sucre",
"Aux sorciers",
"Aux écoures",
"À l'aise",
"À l'aile",
"À l'aîné",
"À l'aime",
"À l'aile de",
"À l'air de",
"À la vitesse",
"À l'idée",
"moi qui cours",
"sur la cours",
"suis en cours",
"Essuie-tout",
"mojito"
]
    FileName =  ['paronyms']*len(Sentenses)
    """ 



    DirTosave = 'TTS_TUITO_New'
    isExist = os.path.exists(DirTosave)
    if not isExist:
        os.makedirs(DirTosave)

    #speed = np.arange (0.8, 2, 0.1)


    voice_pitch_range = [-5, 5]
    voice_speed_range = [1, 1]

    # Test 
    if __name__ == "__main__":
     
      # generating rooms names
      for cpt, Sentense in enumerate(Sentenses):
        Dir_Sentence = os.path.join(DirTosave,FileName[cpt])
        isExist = os.path.exists(Dir_Sentence)
        if not isExist:
            os.makedirs(Dir_Sentence)   
        for A in range(1):
            conf = config_l 
            print(conf) 
            for sp in tqdm(range(100)):
                    voice_pitch = round((voice_pitch_range[1] - voice_pitch_range[0]) * random.random() + voice_pitch_range[0], 2)
                    voice_speed = round((voice_speed_range[1] - voice_speed_range[0]) * random.random() + voice_speed_range[0], 2)                
                    if 'fr-FR' in conf['name']:
                        try:
                            tts_audio_generationfemale(text=Sentense, language_code='fr-FR', name=conf['name'], pitch=voice_pitch, audio_path= Dir_Sentence+'/'+str(uuid.uuid4())+'.wav', speed = voice_speed)
                        except:
                            tts_audio_generationmale(text=Sentense, language_code='fr-FR', name=conf['name'], pitch=voice_pitch, audio_path= Dir_Sentence+'/'+str(uuid.uuid4())+'.wav', speed = voice_speed)
                    else:
                        try:
                            tts_audio_generationfemale(text=Sentense, language_code='fr-CA', name=conf['name'], pitch=voice_pitch, audio_path= Dir_Sentence+'/'+str(uuid.uuid4())+'.wav', speed = voice_speed)
                        except:
                            tts_audio_generationmale(text=Sentense, language_code='fr-CA', name=conf['name'], pitch=voice_pitch, audio_path= Dir_Sentence+'/'+str(uuid.uuid4())+'.wav', speed = voice_speed)            




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type = int ,default=0, help="the desired config.")
    args = parser.parse_args()
    main(args)
