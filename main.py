import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import wandb

from models.graph_learning import PointCloudGraphEnsemble, PointCloudFeatLearning, MLP
from argparse import ArgumentParser

import gc
gc.enable()

# Define the parameters using parser args
parser = ArgumentParser(description="Pointcloud net")
parser.add_argument('--raw_dir', type=str, default = 'melanoma_data_full', help="Directory where the raw data is stored")
parser.add_argument('--full', action='store_true', help="Directory where the raw data is stored")
parser.add_argument('--num_weights', type=int, default=2, help="Number of weights")
parser.add_argument('--threshold', type=float, default= 5e-5, help="Threshold for creating the graph")
parser.add_argument('--hidden_dim', type=int, default= 50, help="Hidden dim for the MLP")
parser.add_argument('--num_layers', type=int, default= 3, help="Number of MLP layers")
parser.add_argument('--lr', type=float, default= 0.03, help="Learnign Rate")
parser.add_argument('--wd', type=float, default= 3e-3, help="Weight decay")
parser.add_argument('--num_epochs', type=int, default= 100, help="Number of epochs")
parser.add_argument('--batch_size', type=int, default= 128, help="Batch size")
parser.add_argument('--gpu', type=int, default= 0, help="GPU index")
args = parser.parse_args()


wandb.init(project='pointcloud-net', 
                 config = vars(args))

if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'


def test(model, mlp, labels, loader):
    model.eval()
    mlp.eval()
    correct = 0
    total = 0                   
    with torch.no_grad():
        for idx in loader:
            X = model(idx, 0.001)
            logits = mlp(X)
            preds = torch.argmax(logits, dim=1)
            correct += torch.sum(preds == labels[idx]).float()
            total += len(idx)
    return (correct*100)/total
    
def train(model, mlp):
    opt = torch.optim.AdamW(list(model.parameters())+list(mlp.parameters()), lr = args.lr, weight_decay = args.wd)
    train_idx, test_idx = train_test_split(np.arange(len(model.labels)), test_size=0.2, stratify=model.labels)
    train_idx = torch.LongTensor(train_idx).to(args.device)
    test_idx = torch.LongTensor(test_idx).to(args.device)
    train_loader = DataLoader(train_idx, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_idx, batch_size=args.batch_size)
    labels = torch.LongTensor(model.labels).to(args.device)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_acc = 0
    for k in range(len(model.graph_feat.alphas)):
        for d in range(len(model.graph_feat.alphas[k])):
            wandb.log({f'Alpha{k}_{d}':model.graph_feat.alphas[k][d].item()}, step=0)
    train_acc = test(model, mlp, labels, train_loader)
    test_acc = test(model, mlp, labels, test_loader)
    wandb.log({'Train acc':train_acc.item(), 'Test acc':test_acc.item()}, step=0)
    if test_acc > best_acc:
            best_acc = test_acc
    with tqdm(range(args.num_epochs)) as tq:
        for e, epoch in enumerate(tq):
            correct_train = 0
            t_loss = 0
            model.train()
            mlp.train()
            for idx in train_loader:
                opt.zero_grad()
                
                X = model(idx, 0.001)
                logits = mlp(X)
                preds = torch.argmax(logits, dim=1)
                correct_train += torch.sum(preds == labels[idx]).float() 
                loss = loss_fn(logits, labels[idx])*400
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        wandb.log({f"{name}.grad": param.grad.norm()}, step=epoch+1)
                opt.step()
                t_loss += loss.item()
                del(X, logits, loss, preds)
                torch.cuda.empty_cache()
                gc.collect()
                    
            train_acc = test(model, mlp, labels, train_loader)
            test_acc = test(model, mlp, labels, test_loader)
            wandb.log({'Loss':t_loss, 'Train acc':train_acc.item(), 'Test acc':test_acc.item()}, step=epoch+1)
            for k in range(len(model.graph_feat.alphas)):
                for d in range(len(model.graph_feat.alphas[k])):
                    wandb.log({f'Alpha{k}_{d}':model.graph_feat.alphas[k][d].item()}, step=epoch+1)
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f'bestalpha_{args.num_weights}')
                torch.save(mlp.state_dict(), f'bestmlp_{args.num_weights}')      
    
            tq.set_description("Train acc = %.4f, Test acc = %.4f, Best acc = %.4f" % (train_acc.item(), test_acc.item(), best_acc))
            
if __name__ == '__main__':
    model = PointCloudFeatLearning(args.raw_dir, args.full, args.num_weights, args.threshold, args.device).to(args.device)
    mlp = MLP(model.input_dim, args.hidden_dim, model.num_labels, args.num_layers).to(args.device)
    torch.save(model.state_dict(), f'bestalpha_{args.num_weights}')
    torch.save(mlp.state_dict(), f'bestmlp_{args.num_weights}')      
    train(model, mlp)
    model.load_state_dict(torch.load(f'bestalpha_{args.num_weights}'))
    mlp.load_state_dict(torch.load(f'bestmlp_{args.num_weights}'))
    torch.save(model.graph_feat.alphas, f'bestweights_{args.num_weights}.pt')
    print(f"Best accuracy : {test(model, mlp, labels, test_loader)}")