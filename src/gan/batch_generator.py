import numpy as np
from scipy.io import wavfile
import os


class BatchGenerator:
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        sampling_frequency, sound = wavfile.read(dir_path + "/../../data/claude-debussy-clair-de-lune.wav")
        normalized_sound = 0.5 * (sound - np.mean(sound)) / np.std(sound)
        cut_samples = len(normalized_sound) // 100
        self.sound = normalized_sound[cut_samples:-cut_samples]

    def format(self, sound):
        return np.expand_dims(np.array(sound), axis=2).astype(np.float32)

    def draw_batch(self, window_len, batch_size):
        sample_starts = np.random.randint(len(self.sound) - 2*window_len, size=batch_size)
        sound_prev = [self.sound[sample_start:sample_start+window_len] for sample_start in sample_starts]
        sound_curr = [self.sound[sample_start+window_len:sample_start+2*window_len] for sample_start in sample_starts]
        return self.format(sound_prev), self.format(sound_curr)

if __name__ == "__main__":
    bg = BatchGenerator()

    sound_prev, sound_curr = bg.draw_batch(1024, 10)

    print(sound_prev.shape)


