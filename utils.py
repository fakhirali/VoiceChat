import numpy as np
import pyaudio
import audioop
from multiprocessing import Value


def is_interupted(is_interrupted, seconds_to_listen=10):
    '''

    :param is_interrupted: multiprocessing value
    :param seconds_to_listen: how long to listen for interruption
    :return: np array of audio
    '''
    seconds_spoken = 0.5  # changing this might make the convo more natural
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  # make sure this is 1
    RATE = 16000
    RECORD_SECONDS = seconds_to_listen
    WAVE_OUTPUT_FILENAME = "user.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* listening for interuption")

    frames = []

    started = False
    interruption = False
    one_second_iters = int(RATE / CHUNK)
    spoken_iters = 0

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        print(spoken_iters)
        data = stream.read(CHUNK)
        frames.append(data)
        rms = audioop.rms(data, p.get_sample_size(FORMAT))
        decibel = 20 * np.log10(rms)
        if not started and decibel > 50:
            started = True

        if started and decibel > 50:
            spoken_iters += 1

        # if started and decibel < 50:
        #     spoken_iters = 0

        if spoken_iters >= one_second_iters * seconds_spoken:
            interruption = True
            break

    if interruption:
        is_interrupted.value = 1
        print("* interrupted")
    else:
        is_interrupted.value = 2
        print("* not interrupted")

    # creating a np array from buffer
    frames = np.frombuffer(b''.join(frames), dtype=np.int16)

    # normalization see https://discuss.pytorch.org/t/torchaudio-load-normalization-question/71470
    frames = frames / (1 << 15)

    return frames.astype(np.float32)



def record_n_seconds(seconds_to_listen=1):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  # make sure this is 1
    RATE = 16000
    RECORD_SECONDS = seconds_to_listen

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    # print("* recording")

    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    # print("* done recording")

    # creating a np array from buffer
    frames = np.frombuffer(b''.join(frames), dtype=np.int16)

    # normalization see https://discuss.pytorch.org/t/torchaudio-load-normalization-question/71470
    frames = frames / (1 << 15)

    return frames.astype(np.float32)



def record(silence_seconds):
    seconds_silence = silence_seconds  # changing this might make the convo more natural
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  # make sure this is 1
    RATE = 16000
    RECORD_SECONDS = 100
    WAVE_OUTPUT_FILENAME = "user.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    started = False
    one_second_iters = int(RATE / CHUNK)
    silent_iters = 0

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
        rms = audioop.rms(data, p.get_sample_size(FORMAT))
        decibel = 20 * np.log10(rms)
        if not started and decibel > 50:
            started = True

        if started and decibel < 50:
            silent_iters += 1

        if started and decibel > 50:
            silent_iters = 0

        if silent_iters >= one_second_iters * seconds_silence:
            break

    print("* done recording")

    # creating a np array from buffer
    frames = np.frombuffer(b''.join(frames), dtype=np.int16)

    # normalization see https://discuss.pytorch.org/t/torchaudio-load-normalization-question/71470
    frames = frames / (1 << 15)

    return frames.astype(np.float32)
