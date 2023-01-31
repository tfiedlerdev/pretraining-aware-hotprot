import torch
from torch.utils.data import DataLoader
import optuna
import time
import copy
from torch import nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from pathlib import Path
from thermostability.thermo_pregenerated_dataset import ThermostabilityPregeneratedDataset
from thermostability.hotinfer_pregenerated import HotInferPregeneratedFC
from tqdm.notebook import tqdm
import sys
from thermostability.thermo_pregenerated_dataset import zero_padding, zero_padding700
import wandb
import argparse

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache() 
    
cpu = torch.device("cpu")

torch.cuda.list_gpu_processes()

train_ds = ThermostabilityPregeneratedDataset('train.csv')
eval_ds = ThermostabilityPregeneratedDataset('val.csv')


dataloaders = {
    "train": DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, collate_fn=zero_padding700),
    "val": DataLoader(eval_ds, batch_size=16, shuffle=True, num_workers=4, collate_fn=zero_padding700)
}

dataset_sizes = {"train": len(train_ds),"val": len(eval_ds)}


def train_model(model, optimizer, criterion, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())

    best_epoch_loss = sys.float_info.max
    losses = []
    batchEnumeration = []
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
         

            # Iterate over data.
            for idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    
                    loss = criterion(outputs,labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if not torch.isnan(loss):
                            loss.backward()
                            threshold = 10
                            for p in model.parameters():
                                if p.grad != None:
                                    if p.grad.norm() > threshold:
                                        torch.nn.utils.clip_grad_norm_(p, threshold)
                            optimizer.step()
                        if torch.isnan(loss).any():
                            print(f"Nan loss: {torch.isnan(loss)}| Loss: {loss}| inputs: {inputs}")
                # statistics
                batch_size = inputs.size(0)
                batch_loss = loss.item() * batch_size
                losses.append(batch_loss)
                batchEnumeration.append(batchEnumeration[-1]+1 if len(batchEnumeration)>0 else 0)

                running_loss += batch_loss
               
            
                if idx % 10 == 0:
                    batch_size = inputs.size(0)
                    tqdm.write("Epoch: [{}/{}], Batch: [{}/{}], loss: {:.6f}".format(
                        epoch,
                        num_epochs,
                        idx + 1,
                        len(dataloaders[phase]),
                        batch_loss / float(batch_size)
                        ), end="\r")
                    
                    
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            
            print(f'{phase} Loss: {epoch_loss:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                wandb.log({"mse_loss": epoch_loss})
        print()


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_epoch_loss:4f}')
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_epoch_loss

def run_train_experiment(config: dict = None):
    with wandb.init(config=config):
        config = wandb.config
        model = HotInferPregeneratedFC(num_hidden_layers=config['model_hidden_layers'], first_hidden_size=config['model_first_hidden_units'])
        model.to(device)
        wandb.watch(model)
        criterion = nn.MSELoss()

        optimizer_ft = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        model, score = train_model(model, optimizer_ft, criterion, exp_lr_scheduler, num_epochs=config["epochs"])

        return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--model_hidden_layers", type=float, required=True)
    parser.add_argument("--model_first_hidden_units", type=float, required=True)
    parser.add_argument("--epochs", type=float, required=True)
    args = parser.parse_args()

    run_train_experiment(config=vars(args))