import pyautogui
import pyaudio
import numpy as np
import librosa
import librosa.feature

note_key_map = {
    "A-1": "a",
    "B-1": "b",
    "C-1": "c",
    "D-1": "d",
    "E-1": "e",
    "F-1": "f",
    "G-1": "g",
}


def detect_note(audio_data, sample_rate):
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    most_prominent_note = np.argmax(np.mean(chroma, axis=1))
    print(most_prominent_note)

    return most_prominent_note


def simulate_key_press(key):
    pyautogui.press(key)


def main():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, input=True, frames_per_buffer=2048)

    try:
        while True:
            audio_data = np.frombuffer(stream.read(2048), dtype=np.float32)
            detected_note_index = detect_note(audio_data, 44100)
            note = librosa.midi_to_note(detected_note_index)
            print(note)
            if note in note_key_map:
                key = note_key_map[note]
                simulate_key_press(key)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
