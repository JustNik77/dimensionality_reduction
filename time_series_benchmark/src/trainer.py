import torch
import numpy as np
from tqdm import tqdm

from utils import moving_average

cuda = "cuda:1"
device = torch.device(cuda) if torch.cuda.is_available() else torch.device("cpu")

def fit(model, optimizer, loss_fn, metric_fn, n_epochs, dataloader_train, dataloader_val, dataloader_test, desc=None):
    
    history = np.zeros((n_epochs, 4, 3))
    
    model.to(device)
    
    pbar = tqdm(range(n_epochs), desc=desc, bar_format="{desc:<17.17}{percentage:3.0f}%|{bar:3}{r_bar}")
    for epoch_idx in pbar:
        
        # train
        model.train()
        
        loss_batches = np.zeros((len(dataloader_train), 4))
        for i, (X, Y) in enumerate(dataloader_train):
            X, Y = X.to(device), Y.to(device)
            
            Y_hat = model(X)
            loss_batch = loss_fn(Y_hat, Y)
            
            loss_batch.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            with torch.no_grad():
                loss_batches[i,0] = loss_batch.detach() # MSE
                loss_batches[i,1] = metric_fn(Y_hat, Y).detach() # MAE
        history[epoch_idx,:,0] = loss_batches.mean(axis=0)
        
        # val/test
        model.eval()

        loss_batches = np.zeros((len(dataloader_val), 4))
        for i, (X, Y) in enumerate(dataloader_val):
            X, Y = X.to(device), Y.to(device)
            Y_hat = model(X)
            loss_batches[i,0] = loss_fn(Y_hat, Y).detach() # MSE
            loss_batches[i,1] = metric_fn(Y_hat, Y).detach() # MAE
        history[epoch_idx,:,1] = loss_batches.mean(axis=0)

        loss_batches = np.zeros((len(dataloader_test), 4))
        for i, (X, Y) in enumerate(dataloader_test):
            X, Y = X.to(device), Y.to(device)
            Y_hat = model(X)
            loss_batches[i,0] = loss_fn(Y_hat, Y).detach() # MSE
            loss_batches[i,1] = metric_fn(Y_hat, Y).detach() # MAE
        history[epoch_idx,:,2] = loss_batches.mean(axis=0)
        
        pbar.set_postfix_str("t={:.4f}, t*={:.4f}, v={:.4f}, v*={:.4f}, t={:.4f}, t*={:.4f}, t@v*={:.4f}, t@v**={:.4f}".format(
            history[epoch_idx,0,0], # t
            np.min(history[:epoch_idx+1,0,0]), # v*
            history[epoch_idx,0,1], # v
            np.min(history[:epoch_idx+1,0,1]), # v*
            history[epoch_idx,0,2], # test
            np.min(history[:epoch_idx+1,0,2]), # test*
            history[np.argmin(history[:epoch_idx+1,0,1]),0,2], # test@v*
            history[np.argmin(moving_average(history[:epoch_idx+1,0,1])),0,2] # test@v**
        ))
        
    return model, history