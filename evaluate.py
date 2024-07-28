import numpy as np
import datetime
import json
import argparse
import torch
import torch.nn as nn

from model.conformer import RoPEConformer
from data_loader import MyDataLoader
from utils import int2text, greedy_decode, calculate_wer

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Evaluate trained speech recognition model.')
parser.add_argument('--config', type=str, required=True, help='Path to the config file')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
args = parser.parse_args()

# Load config.json
with open(args.config, 'r') as f:
    config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset_config = config.get('dataset')
audio_config = config.get('audio')

data_loader = MyDataLoader(audio_config)
# Load datasets
#==========================================***===========================================
test_dataset = data_loader.load_librispeech_datasets(dataset_config['librispeech_path'], 
                                                     dataset_config['test_subsets'])
valid_dataset = data_loader.load_librispeech_datasets(dataset_config['librispeech_path'],
                                                      dataset_config['valid_subsets'])

# Get DataLoader
#==========================================***===========================================
test_loader = data_loader.get_dataloader(test_dataset, 
                                          batch_size=1, 
                                          shuffle=False)
valid_loader = data_loader.get_dataloader(valid_dataset, 
                                          batch_size=1, 
                                          shuffle=False)

# Model, Loss, and Optimizer
#==========================================***===========================================
model_params = config.get('acoustic_model')
model = RoPEConformer(**model_params).to(device)

# Load model checkpoint
#==========================================***===========================================
checkpoint = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(device)

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    wer_list = []
    
    with torch.no_grad():
        for spectrograms, labels, input_lengths, label_lengths in dataloader:
            spectrograms, labels, input_lengths, label_lengths = (spectrograms.to(device),
                                labels.to(device),
                                input_lengths.to(device),
                                label_lengths.to(device))
            output, output_lengths = model(spectrograms, input_lengths)
            loss = criterion(output.transpose(0, 1), labels, output_lengths, label_lengths)
            total_loss += loss.item()
            # Greedy decoding
            decoded_preds = greedy_decode(torch.argmax(output, dim=2).squeeze(0).tolist())
            decoded_preds_text = int2text(decoded_preds)
            labels_text = int2text(labels.cpu().numpy().flatten())
            # Calculate WER
            wer = calculate_wer(labels_text, decoded_preds_text)
            wer_list.append(wer)
    
    avg_loss = total_loss / len(dataloader)
    avg_wer = np.mean(wer_list)
    
    return avg_loss, avg_wer

# Evaluate on validation and test sets
valid_loss, valid_wer = evaluate(model, valid_loader, criterion, device)
test_loss, test_wer = evaluate(model, test_loader, criterion, device)

print(f'Validation Loss: {valid_loss:.8f}, Validation WER: {valid_wer:.8f}')
print(f'Test Loss: {test_loss:.8f}, Test WER: {test_wer:.8f}')
