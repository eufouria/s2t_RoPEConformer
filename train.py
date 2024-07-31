import datetime
import json
import argparse
import torch
import torch.nn as nn

from model.conformer import RoPEConformer
from data_loader import MyDataLoader

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a speech recognition model.')
parser.add_argument('--config', type=str, required=True, help='Path to the config file')
args = parser.parse_args()

# Load config.json
with open(args.config, 'r') as f:
    config = json.load(f)

seed = config.get('seed')
print('Seeding with {}'.format(seed))
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset_config = config.get('dataset')
training_config = config.get('training')
audio_config = config.get('audio')

data_loader = MyDataLoader(audio_config)

# Load datasets
#==========================================***===========================================
train_dataset = data_loader.load_librispeech_datasets(dataset_config['librispeech_path'], 
                                                      dataset_config['train_subsets'])

# Get DataLoader
#==========================================***===========================================
train_loader = data_loader.get_dataloader(train_dataset, 
                                          batch_size=training_config['batch_size'], 
                                          shuffle=True, 
                                          num_workers=training_config['num_workers'])


# Model, Loss, and Optimizer
#==========================================***===========================================
model_params = config.get('acoustic_model')
model = RoPEConformer(**model_params).to(device)


# Optimizer
#==========================================***===========================================
optimizer = torch.optim.AdamW(model.parameters(), lr=training_config['learning_rate'])

# Scheduler
#==========================================***===========================================
scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=training_config['max_lr'],
        steps_per_epoch=int(len(train_loader)),
        epochs=training_config['epochs'],
        anneal_strategy=training_config['anneal_strategy']
    )

# Loss function
#==========================================***===========================================
criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(device) # blank=0 is the index for unknown characters


# training loop
#==========================================***===========================================
def train(model, dataloader, criterion, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    batch_loss = []
    for spectrograms, labels, input_lengths, label_lengths in dataloader:
        spectrograms, labels, input_lengths, label_lengths = (spectrograms.to(device),
                                labels.to(device),
                                input_lengths.to(device),
                                label_lengths.to(device))
        optimizer.zero_grad()
        output = model(spectrograms, input_lengths)
        output, output_lengths = model(spectrograms, input_lengths)
        loss = criterion(output.transpose(0, 1), labels, output_lengths, label_lengths)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        batch_loss.append(loss.item())
    return total_loss / len(dataloader), batch_loss


# Train model
#==========================================***===========================================
n_epochs = training_config['epochs']
batch_losses = []
for epoch in range(n_epochs):
    start = datetime.datetime.now()
    loss, batch_loss = train(model, train_loader, criterion, optimizer, device, scheduler)
    batch_losses.append(batch_loss)
    print(f'Epoch {epoch}, avg. Loss: {round(loss, 8)}, Time Elapsed: {datetime.datetime.now()-start}')
    if epoch % 5 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'batch_loss':batch_loss
        }
        torch.save(checkpoint, training_config["checkpoint_path"].format(epoch))
