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
        # Train
        model.train()
        loss_batches = np.zeros((len(dataloader_train), 4))
        
        for i, batch in enumerate(dataloader_train):
            x_center = batch['x_center'].float().to(device)
            x_neighbors = batch['x_neighbors'].float().to(device)
            
            # Forward pass
            output = model(x_center, x_neighbors)
            loss_batch = output["loss"]
            mse_loss = output["mse_loss"]
            
            # Backward pass
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            
            # Сохраняем значения лоссов
            with torch.no_grad():
                loss_batches[i,0] = mse_loss.item()  # MSE
                loss_batches[i,1] = loss_batch.item()  # Neighborhood loss
                
        history[epoch_idx,:,0] = loss_batches.mean(axis=0)
        
        # Validation
        model.eval()
        with torch.no_grad():
            loss_batches = np.zeros((len(dataloader_val), 4))
            
            for i, batch in enumerate(dataloader_val):
                x_center = batch['x_center'].float().to(device)
                x_neighbors = batch['x_neighbors'].float().to(device)
                
                output = model(x_center, x_neighbors)
                mse_loss = output["mse_loss"]
                neighbor_loss = output["loss"]
                
                loss_batches[i,0] = mse_loss.item()  # MSE
                loss_batches[i,1] = neighbor_loss.item()  # Neighborhood loss
                
            history[epoch_idx,:,1] = loss_batches.mean(axis=0)
            
            # Test
            loss_batches = np.zeros((len(dataloader_test), 4))
            
            for i, batch in enumerate(dataloader_test):
                x_center = batch['x_center'].float().to(device)
                x_neighbors = batch['x_neighbors'].float().to(device)
                
                output = model(x_center, x_neighbors)
                mse_loss = output["mse_loss"]
                neighbor_loss = output["loss"]
                
                loss_batches[i,0] = mse_loss.item()  # MSE
                loss_batches[i,1] = neighbor_loss.item()  # Neighborhood loss
                
            history[epoch_idx,:,2] = loss_batches.mean(axis=0)
        
        pbar.set_postfix_str("t={:.4f}, t*={:.4f}, v={:.4f}, v*={:.4f}, t={:.4f}, t*={:.4f}, t@v*={:.4f}, t@v**={:.4f}".format(
            history[epoch_idx,0,0], # t (train mse)
            np.min(history[:epoch_idx+1,0,0]), # t* (min train mse)
            history[epoch_idx,0,1], # v (val mse)
            np.min(history[:epoch_idx+1,0,1]), # v* (min val mse)
            history[epoch_idx,0,2], # test mse
            np.min(history[:epoch_idx+1,0,2]), # min test mse
            history[np.argmin(history[:epoch_idx+1,0,1]),0,2], # test@v* (test at best val)
            history[np.argmin(moving_average(history[:epoch_idx+1,0,1])),0,2] # test@v** (test at best smoothed val)
        ))
    
    return model, history
