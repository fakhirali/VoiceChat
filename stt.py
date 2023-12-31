from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torchaudio.functional as F
import torch
from utils import record, record_n_seconds
import numpy as np
print(); print()



class Ear:
    def __init__(self, model_id='openai/whisper-base.en', device='cpu', silence_seconds=1):
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self.device = device
        self.model.to(device)
        self.silence_seconds = silence_seconds

    @torch.no_grad()
    def transcribe(self, audio):
        input_features = self.processor(audio, sampling_rate=16_000, return_tensors="pt").input_features.to(self.device) 
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return ' '.join(transcription)
    

    def listen(self):
        audio = record(self.silence_seconds)
        text = self.transcribe(audio)
        print(text)
        return text

    def listen_for_interruption(self, is_interrupted, seconds_to_listen=1):
        audio = False
        for i in range(0, int(seconds_to_listen)):
            new_audio = record_n_seconds(seconds_to_listen=1)
            if audio is False:
                audio = new_audio
            else:
                audio = np.hstack((audio, new_audio))
            print(audio.shape)
            text = self.transcribe(audio)
            print(text)
            if text != '':
                print('interrupted:', text)
                is_interrupted.value = 1
                return
        is_interrupted.value = 2


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ear = Ear(device=device)

    audio, sr = torchaudio.load('media/abs.wav')
    audio = F.resample(audio, sr, 16_000)[0]
    text = ear.transcribe(audio)
    print(text)

    text = ear.listen()
    print(text)

