from transformers import VitsModel, AutoTokenizer
import sounddevice as sd
import torch
from utils import is_interupted
from multiprocessing import Value
from threading import Thread
from stt import Ear

print();
print()


class Mouth:
    def __init__(self, model_id='kakao-enterprise/vits-vctk', speaker_id=0, device='cpu'):
        self.model = VitsModel.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = device
        self.model.to(device)
        self.speaker_id = speaker_id


    @torch.no_grad()
    def say_with_interruption(self, text, ear):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = inputs.to(self.device)
        output = self.model(**inputs, speaker_id=self.speaker_id).waveform[0].to('cpu')
        # sd.play(output, samplerate=self.model.config.sampling_rate)

        ## break on interruption
        isInterupted = Value('i', 0)
        seconds_of_audio = len(output) / self.model.config.sampling_rate
        p = Thread(target=ear.listen_for_interruption, args=(isInterupted, seconds_of_audio))
        p.start()
        while True:
            if isInterupted.value == 1:
                sd.stop()
                break
            elif isInterupted.value == 2:
                break
        p.join()
        # sd.wait()


    @torch.no_grad()
    def say(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = inputs.to(self.device)
        output = self.model(**inputs, speaker_id=self.speaker_id).waveform[0].to('cpu')
        sd.play(output, samplerate=self.model.config.sampling_rate)
        sd.wait()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mouth = Mouth(device=device, speaker_id=4)

    text = "If there's one thing that makes me nervous about the future of self-driving cars, it's that they'll replace human drivers.\nI think there's a huge opportunity to make human-driven cars safer and more efficient. There's no reason why we can't combine the benefits of self-driving cars with the ease of use of human-driven cars."
    print(text)
    # mouth.say(text)
    ear = Ear(device=device, silence_seconds=1)
    ear.listen()
    mouth.say_with_interruption(text, ear)
    sd.wait()
