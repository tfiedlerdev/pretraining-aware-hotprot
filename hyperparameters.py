import torch
from torch.utils.data import DataLoader
import optuna
import time
import copy
from torch import nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from pathlib import Path
import mlflow
from optuna.integration.mlflow import MLflowCallback
from thermostability.thermo_pregenerated_dataset import ThermostabilityPregeneratedDataset
from thermostability.hotinfer_pregenerated import HotInferPregeneratedLSTM
from tqdm.notebook import tqdm
import sys
from thermostability.thermo_pregenerated_dataset import zero_padding

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.empty_cache() 
    
cpu = torch.device("cpu")

torch.cuda.list_gpu_processes()

train_ds = ThermostabilityPregeneratedDataset('train.csv')
eval_ds = ThermostabilityPregeneratedDataset('val.csv')


dataloaders = {
    "train": DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, collate_fn=zero_padding),
    "val": DataLoader(eval_ds, batch_size=32, shuffle=True, num_workers=4, collate_fn=zero_padding)
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

        print()


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_epoch_loss:4f}')
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_epoch_loss

YOUR_TRACKING_URI = "http://127.0.0.1:5000"
mlflc = MLflowCallback(
    tracking_uri=YOUR_TRACKING_URI,
    metric_name="metric_score"
)
@mlflc.track_in_mlflow()
def optimize_thermostability(trial):    
    params = {
        'model_learning_rate': trial.suggest_float('model_learning_rate', 0.01, 0.75, step=0.05),
        'model_hidden_units': trial.suggest_int('model_hidden_units', 64, 640, step=64),
        'model_hidden_layers': trial.suggest_int('model_hidden_layers', 1, 4, step=1)
    }
    
    model = HotInferPregeneratedLSTM(
        params['model_hidden_units'],
        params['model_hidden_layers'],
    )
    model.to(device)
    
    criterion = nn.MSELoss()

    optimizer_ft = torch.optim.SGD(model.parameters(), lr=params['model_learning_rate'], momentum=0.9)
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model, score = train_model(model, optimizer_ft, criterion, exp_lr_scheduler, num_epochs=1)

    mlflow.log_params(params)
    mlflow.pytorch.log_model(model, "model")
    # candidate_model_uri = mlflow.pytorch.log_model(model).model_uri
    mlflow.log_metric("score", score)
    return score

# minimize or maximize
study = optuna.create_study(direction="minimize", study_name="thermostability-hyperparameters") # maximise the score during tuning
study.optimize(optimize_thermostability, n_trials=20) # run the objective function 100 times

print(study.best_trial) # print the best performing pipeline