import argparse
import datetime
import json
from loguru import logger
import numpy as np
import torch
import torch.nn as nn

from data_loader import MyDataLoader
from lr_scheduler import NoamAnnealing
from model.conformer import RoPEConformer
from utils import calculate_wer, int2text, get_n_params, greedy_decode 

# Configure loguru
logger.add("training_logs_{time}.log", format="{time} {level} {message}", level="INFO")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a speech recognition model.')
parser.add_argument('--config', type=str, required=True, help='Path to the config file')
args = parser.parse_args()

# Load config.json
with open(args.config, 'r') as f:
    config = json.load(f)

seed = config.get('seed')
logger.info('Seeding with {}'.format(seed))
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using {torch.cuda.device_count()} {device}")

dataset_config = config.get('dataset')
training_config = config.get('training')
audio_config = config.get('audio')

data_loader = MyDataLoader(audio_config)

# Load train datasets
#==========================================***===========================================
train_dataset = data_loader.load_librispeech_datasets(dataset_config['librispeech_path'], 
                                                      dataset_config['train_subsets'])

# Get Train DataLoader
#==========================================***===========================================
train_loader = data_loader.get_dataloader(train_dataset, 
                                          batch_size=training_config['batch_size'], 
                                          shuffle=True, 
                                          num_workers=training_config['num_workers'])
logger.info(f"Train DataLoader Size: {len(train_loader)}")

# Load validation datasets
#==========================================***===========================================
valid_dataset = data_loader.load_librispeech_datasets(dataset_config['librispeech_path'], 
                                                      dataset_config['valid_subsets'])

# Get Validation DataLoader
#==========================================***===========================================
valid_loader = data_loader.get_dataloader(valid_dataset, 
                                          batch_size=1, 
                                          shuffle=False, 
                                          num_workers=training_config['num_workers'])
logger.info(f"Validation DataLoader Size: {len(valid_loader)}")


# Model, Loss, and Optimizer
#==========================================***===========================================
model_params = config.get('acoustic_model')
logger.info(f"HyperParams: {model_params}")
model = RoPEConformer(**model_params).to(device)
logger.info(f"Total params: {get_n_params(model)}")

# Use DataParallel if multiple GPUs are available
#==========================================***===========================================
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model.to('cuda')

# Optimizer
#==========================================***===========================================
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=training_config['lr'],
    betas=training_config['betas'],
    weight_decay=training_config['weight_decay']
)

# Scheduler
#==========================================***===========================================
d_model = model_params['enc_hidden_dims']
scheduler = NoamAnnealing(
    optimizer,
    d_model=d_model,
    warmup_steps=training_config['warmup_steps'],
    min_lr=training_config['min_lr'],
)


# Loss function
#==========================================***===========================================
criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).to(device) # blank=0 is the index for blank characters


# training loop   
#==========================================***===========================================
def train(model, dataloader, criterion, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    batch_loss = []

    for idx, (spectrograms, labels, input_lengths, label_lengths) in enumerate(dataloader):
        if idx % 100 == 0 and idx > 0:
            current_lr = scheduler.get_last_lr()[0]  # Assuming single learning rate
            logger.info(f'  ||  Batch {idx}, \
            Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, \
            Avg. Loss: {round(total_loss/(idx+1), 6)}, \
            LR: {round(current_lr, 8)}')
        
        optimizer.zero_grad()
        spectrograms, labels, input_lengths, label_lengths = (
            spectrograms.to(device),
            labels.to(device),
            input_lengths.to(device),
            label_lengths.to(device)
        )

        output, output_lengths = model(spectrograms, input_lengths)
        loss = criterion(output.transpose(0, 1), labels, output_lengths, label_lengths)

        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        batch_loss.append(loss.item())
    return total_loss / len(dataloader), batch_loss


# Validate model
#==========================================***===========================================
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    wer_list = []
    with torch.no_grad():
        for idx, (spectrograms, labels, input_lengths, label_lengths) in enumerate(dataloader):
            spectrograms, labels, input_lengths, label_lengths = (
                spectrograms.to(device),
                labels.to(device),
                input_lengths.to(device),
                label_lengths.to(device)
            )
            
            # Forward pass
            output, output_lengths = model(spectrograms, input_lengths)
            loss = criterion(output.transpose(0, 1), labels, output_lengths, label_lengths)
            total_loss += loss.item()
            
            # Greedy decoding
            decoded_preds = greedy_decode(torch.argmax(output, dim=2).squeeze(0).tolist())
            decoded_preds_text = int2text(decoded_preds)
            labels_text = int2text(labels.cpu().numpy().flatten())
            if idx <= 3:
                logger.info(f"Reference: {labels_text}")
                logger.info(f"Hypothesis: {decoded_preds_text}")
            # Calculate WER
            wer = calculate_wer(labels_text, decoded_preds_text)
            wer_list.append(wer)
    
    avg_loss = total_loss / len(dataloader)
    avg_wer = np.mean(wer_list)
    return avg_loss, avg_wer

# Train model
#==========================================***===========================================
# Early Stopping Parameters
patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

n_epochs = training_config['epochs']
batch_losses = []
for epoch in range(n_epochs):
    if early_stop:
        logger.info(f"Early stopping at epoch {epoch}")
        break
    
    logger.info(f'  Epoch {epoch}')
    start = datetime.datetime.now()
    
    # Train for one epoch
    loss, batch_loss = train(model, train_loader, criterion, optimizer, device, scheduler)
    batch_losses.append(batch_loss)
    
    # Validate after each epoch
    val_loss, wer = validate(model, valid_loader, criterion, device)
    logger.info(f'  Epoch {epoch}, avg. Train Loss: {round(loss, 6)}, \
    avg. Val Loss: {round(val_loss, 6)}, \
    WER: {round(wer, 6)}, \
    Time Elapsed: {datetime.datetime.now()-start}')
    logger.info(f'==========================================***===========================================')

    # Check if validation loss improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # Save the best model
        torch.save(model.state_dict(), training_config["checkpoint_path"].format("best"))
    else:
        epochs_no_improve += 1
    
    # Early stopping
    if epochs_no_improve >= patience:
        early_stop = True

    # Save checkpoint every 5 epochs
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'batch_loss': batch_loss,
        'val_loss': val_loss
    }
    torch.save(checkpoint, training_config["checkpoint_path"].format(epoch))
