import random
import torch
import torchaudio
import torchaudio.transforms as transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class MyDataLoader:
    def __init__(self, audio_config):
        self.char_to_idx = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9, "j": 10,
                   "k": 11, "l": 12, "m": 13, "n": 14, "o": 15, "p": 16, "q": 17, "r": 18, "s": 19,
                   "t": 20, "u": 21, "v": 22, "w": 23, "x": 24, "y": 25, "z": 26, " ": 27, "<UNK>": 28}
        self.audio_config = audio_config

    def text2int(self, text):
        return [self.char_to_idx.get(c, 28) for c in text]  # 28 is the idx for unknown characters

    def spec_augment(self, spectrogram):
        # Frequency masking
        freq_masking = transforms.FrequencyMasking(freq_mask_param=27)
        spectrogram = freq_masking(spectrogram)
        spectrogram = freq_masking(spectrogram)  # Applying twice

        # Adaptive Time masking
        num_time_steps = spectrogram.shape[1]
        pM = 0.04
        pS = 0.04
        num_masks = int(num_time_steps * pM)
        mask_size = int(num_time_steps * pS)
        
        for _ in range(num_masks):
            start = random.randint(0, num_time_steps - mask_size)
            time_masking = transforms.TimeMasking(time_mask_param=mask_size)
            spectrogram = time_masking(spectrogram)
        return spectrogram

    def data_processing(self, data):
        spectrograms, labels, input_lengths, label_lengths = [], [], [], []

        for (waveform, _, utterance, _, _, _) in data:
            mel_transform = transforms.MelSpectrogram(
                sample_rate=self.audio_config['sample_rate'],
                n_mels=self.audio_config['n_mels'],
                n_fft=self.audio_config['n_fft'],
                win_length=self.audio_config['win_length'],
                hop_length=self.audio_config['hop_length']
            )
            spectrogram = mel_transform(waveform).squeeze(0).transpose(0, 1)
            
            # Apply the SpecAugment policy
            spectrogram = self.spec_augment(spectrogram)

            spectrograms.append(spectrogram)
            label = torch.Tensor(self.text2int(utterance.lower()))
            labels.append(label)
            input_lengths.append(spectrogram.shape[0])
            label_lengths.append(len(label))
        
        spectrograms = pad_sequence(spectrograms, batch_first=True).unsqueeze(1)
        labels = pad_sequence(labels, batch_first=True)
        input_lengths = torch.Tensor(input_lengths).int()
        label_lengths = torch.Tensor(label_lengths).int()
        
        return spectrograms, labels, input_lengths, label_lengths

    def get_dataloader(self, dataset, batch_size, shuffle=True, num_workers=0):
        return DataLoader(dataset, 
                          batch_size=batch_size, 
                          shuffle=shuffle, 
                          num_workers=num_workers, 
                          collate_fn=self.data_processing)

    def load_librispeech_datasets(self, dataset_path, subsets):
        datasets = []
        for subset in subsets:
            dataset = torchaudio.datasets.LIBRISPEECH(dataset_path, 
                                                      url=subset, 
                                                      download=True)
            datasets.append(dataset)
        return torch.utils.data.ConcatDataset(datasets)