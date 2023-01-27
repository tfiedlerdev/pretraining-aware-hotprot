
from tqdm.notebook import tqdm
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import random_split
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from thermostability.thermo_pregenerated_dataset import ThermostabilityPregeneratedDataset
from util.telegram import TelegramBot
from thermostability.hotinfer_pregenerated import HotInferPregenerated
from datetime import datetime

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache() 
    
cpu = torch.device("cpu")

torch.cuda.list_gpu_processes()

telegramBot = TelegramBot()
telegramBot.enabled=False

trainSet = ThermostabilityPregeneratedDataset("data/s_s/train/")
valSet = ThermostabilityPregeneratedDataset("data/s_s/val/")

dataloaders = {
    "train": torch.utils.data.DataLoader(trainSet, batch_size=16, shuffle=True, num_workers=4),
    "val": torch.utils.data.DataLoader(valSet, batch_size=16, shuffle=True, num_workers=4)
}

dataset_sizes = {"train": len(trainSet),"val": len(valSet)}
print(dataset_sizes)
print(next(enumerate(dataloaders["train"])))

model = HotInferPregenerated()

model.to(device)

def train_model(model, criterion,optimizer , scheduler, num_epochs=25):
    since = time.time()
    telegramBot.send_telegram(f"===> Starting training ({num_epochs} epochs, {len(trainSet)} train samples, {len(valSet)} val samples)")
    #best_model_wts = copy.deepcopy(model.state_dict())

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
         
            sliding_loss = 0.0
            # Iterate over data.
            for idx, (inputs, labels) in enumerate(dataloaders[phase]):
                #inputs = inputs.to(device)
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
                batch_size = len(inputs)
                batch_loss = loss.item() * batch_size
                losses.append(batch_loss)
                batchEnumeration.append(batchEnumeration[-1]+1 if len(batchEnumeration)>0 else 0)

                running_loss += batch_loss
                sliding_loss += batch_loss
           
                if idx % 10 == 0:
                    tqdm.write("Epoch: [{}/{}], Batch: [{}/{}], loss: {:.6f}".format(
                        epoch,
                        num_epochs,
                        idx + 1,
                        len(dataloaders[phase]),
                        batch_loss / float(batch_size)
                        ), end="\r")
                
                telegramFrequency = 10
                if idx % telegramFrequency == telegramFrequency-1:
                    telegramBot.send_telegram("Epoch: [{}/{}], Batch: [{}/{}], Batch loss: {:.6f}, Total Avg Epoch loss: {:.6f}, Avg loss last {} epochs: {:.6f}".format(
                        epoch,
                        num_epochs,
                        idx + 1,
                        len(dataloaders[phase]),
                        batch_loss / float(batch_size),
                        (running_loss/batch_size)/(idx+1),
                        telegramFrequency,
                        (sliding_loss/batch_size)/telegramFrequency
                        ))
                    sliding_loss = 0.    
                    
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]


            print(f'{phase} Loss: {epoch_loss:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                #best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        telegramBot.send_telegram(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_epoch_loss:4f}')
        # load best model weights
        #model.load_state_dict(best_model_wts)
        return model


criterion = nn.MSELoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

try:
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=1)
except Exception as e: 
    print(e)
    telegramBot.send_telegram(f"Training failed with error message: {str(e)}")                         



def predictDiffs(set="val"):
    with torch.no_grad():
        n = len(dataloaders[set])
        diffs = torch.tensor([])
        for index, (inputs, labels) in enumerate(dataloaders[set]):
            #inputs = inputs.to(device)
            print(f"Infering thermostability for sample {index}/{n}...")
            labels = labels.to(device)
            outputs = model(inputs)

            _diffs = outputs.squeeze(1).sub(labels.squeeze(1)).cpu()
            diffs = torch.cat((diffs, _diffs))
            print("Diff: ", _diffs)
    return diffs
diffs = predictDiffs()

#diffs = np.array([0, 0.1, 0.2,-0.2, -0.8, 0.1])
plt.title("Differences predicted <-> actual thermostability")
plt.hist(diffs, 10)
resultsDir = "results"
now = datetime.now()
time = now.strftime("%d/%m/%Y_%H:%M:%S")
os.makedirs(resultsDir, exist_ok=True)
histFile = f"results/{time}_diffs.png"
plt.savefig(histFile)
telegramBot.send_photo(histFile, f"Differences predicted <-> actual thermostability at {time}")


try: 
    modelPath = os.path.join(resultsDir, f"{time}_model.pth")
    torch.save(model, modelPath)
    telegramBot.send_telegram(f"Model saved at {modelPath}")
except Exception as e:
    telegramBot.send_telegram(f"Saving model failed for reason: {str(e)}")