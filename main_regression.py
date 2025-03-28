import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.read_data import load_data_persistence
import wandb
from torchinfo import summary
from torchviz import make_dot
from models.graph_learning import HiPoNet, MLP
from argparse import ArgumentParser

import gc
gc.enable()

# Define the parameters using parser args
parser = ArgumentParser(description="Pointcloud net")
parser.add_argument('--raw_dir', type=str, default = 'melanoma_data_full', help="Directory where the raw data is stored")
parser.add_argument('--full', action='store_true')
parser.add_argument('--orthogonal', action='store_true')
parser.add_argument('--model', type=str, default = 'graph', help="Type of structure")
parser.add_argument('--num_weights', type=int, default=2, help="Number of weights")
parser.add_argument('--threshold', type=float, default= 0.5, help="Threshold for creating the graph")
parser.add_argument('--hidden_dim', type=int, default= 500, help="Hidden dim for the MLP")
parser.add_argument('--num_layers', type=int, default= 3, help="Number of MLP layers")
parser.add_argument('--lr', type=float, default= 1e-1, help="Learnign Rate")
parser.add_argument('--wd', type=float, default= 3e-3, help="Weight decay")
parser.add_argument('--num_epochs', type=int, default= 20, help="Number of epochs")
parser.add_argument('--batch_size', type=int, default= 128, help="Batch size")
parser.add_argument('--gpu', type=int, default= 0, help="GPU index")
args = parser.parse_args()


wandb.init(project='pointcloud-net-persistence-prediction', 
                 config = vars(args))

if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

loss_fn = torch.nn.MSELoss()

def test(model, mlp, PCs, labels, loader):
    model.eval()
    mlp.eval()
    mse = 0
    total = 0                   
    with torch.no_grad():
        for idx in tqdm(loader):
            X = model([PCs[i].to(args.device) for i in idx], 5)
            preds = mlp(X)
            mse += (loss_fn(preds, labels[idx]) * len(idx))
            total += len(idx)
    return mse*1000/total
    
def train(model, mlp, PCs, labels):
    print(args)
    opt = torch.optim.AdamW(list(model.parameters())+list(mlp.parameters()), lr = args.lr, weight_decay = args.wd)
    train_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2)
    train_idx = torch.LongTensor(train_idx).to(args.device)
    test_idx = torch.LongTensor(test_idx).to(args.device)
    train_loader = DataLoader(train_idx, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_idx, batch_size=args.batch_size)
    labels = labels.to(args.device).float()
    for k in range(len(model.layer.alphas)):
        for d in range(len(model.layer.alphas[k])):
            wandb.log({f'Alpha{k}_{d}':model.layer.alphas[k][d].item()}, step=0)
    train_mse = test(model, mlp, PCs, labels, train_loader)
    best_mse = test(model, mlp, PCs, labels, test_loader)
    wandb.log({'Train MSE':train_mse.item(), 'Test MSE':best_mse.item()}, step=0)
    with tqdm(range(args.num_epochs)) as tq:
        for e, epoch in enumerate(tq):
            t_loss = 0
            model.train()
            mlp.train()
            for idx in (train_loader):
                opt.zero_grad()
                X = model([PCs[i].to(args.device) for i in idx], 5)
                
                # X = model(idx, 0.000001)
                logits = mlp(X)
                loss = loss_fn(logits, labels[idx]) * 1000
                if(args.orthogonal):
                    loss += 0.1*(model.layer.alphas@model.layer.alphas.T - torch.eye(args.num_weights).to(args.device)).square().mean()
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        wandb.log({f"{name}.grad": param.grad.norm()}, step=epoch+1)
                opt.step()
                t_loss += loss.item()
                del(X, logits, loss)
                torch.cuda.empty_cache()
                gc.collect()
                    
            train_mse = test(model, mlp, PCs, labels, train_loader)
            test_mse = test(model, mlp, PCs, labels, test_loader)
            wandb.log({'Loss':t_loss, 'Train MSE':train_mse.item(), 'Test MSE':test_mse.item()}, step=epoch+1)
            for k in range(len(model.layer.alphas)):
                for d in range(len(model.layer.alphas[k])):
                    wandb.log({f'Alpha{k}_{d}':model.layer.alphas[k][d].item()}, step=epoch+1)
            if test_mse < best_mse:
                best_mse = test_mse
                model_path = f"persistence_models/model_{args.num_weights}.pth"

                # torch.save({
                #     'epoch': epoch,  # Save the current epoch number
                #     'model_state_dict': model.state_dict(),
                #     'mlp_state_dict': mlp.state_dict(),
                #     'optimizer_state_dict': opt.state_dict(),
                #     'best_mse': best_mse,
                #     'args': args
                # }, model_path)
    
            tq.set_description("Train MSE = %.4f, Test MSE = %.4f, Best MSE = %.4f" % (train_mse.item(), test_mse.item(), best_mse))
    print(f"Best MSE : {best_mse}")
            
if __name__ == '__main__':
    PCs, labels, num_labels = load_data_persistence(args.raw_dir, args.full)
    model = HiPoNet(args.model, PCs[0].shape[1], args.num_weights, args.threshold, args.device).to(args.device)
    with torch.no_grad():
        input_dim = model([PCs[0].to(args.device)], 10).shape[1]
    mlp = MLP(input_dim, args.hidden_dim, num_labels, args.num_layers).to(args.device)
    model_path = f"persistence_models/model_{args.raw_dir}_{args.num_weights}_{args.model}_{args.orthogonal}.pth"

    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     'mlp_state_dict': mlp.state_dict(),
    #     'best_mse': 0,
    #     'args': args
    # }, model_path)
    
    train(model, mlp, PCs, labels)