import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from models.graph_learning import PointCloudGraphEnsemble, PointCloudFeatLearning, MLP
from argparse import ArgumentParser

import gc
gc.enable()

# Define the parameters using parser args
parser = ArgumentParser(description="Graph Ensemble Layer for Point Clouds")
parser.add_argument('--raw_dir', type=str, default = 'data/melanoma/raw', help="Directory where the raw data is stored")
parser.add_argument('--kernel_type', type=str, choices=['gaussian', 'alpha_decay'], default='gaussian', help="Type of kernel function")
parser.add_argument('--threshold', type=float, default= 5e-5, help="Threshold for creating the graph")
parser.add_argument('--hidden_dim', type=int, default= 300, help="Hidden dim for the MLP")
parser.add_argument('--num_layers', type=int, default= 5, help="Number of MLP layers")
parser.add_argument('--lr', type=float, default= 1e-2, help="Learnign Rate")
parser.add_argument('--wd', type=float, default= 3e-4, help="Weight decay")
parser.add_argument('--num_epochs', type=int, default= 5, help="Number of epochs")
parser.add_argument('--batch_size', type=int, default= 16, help="Batch size")
parser.add_argument('--gpu', type=int, default= 0, help="GPU index")


args = parser.parse_args()

if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

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
    with tqdm(range(args.num_epochs)) as tq:
        for epoch in enumerate(tq):
            correct_train = 0
            t_loss = 0
            for idx in train_loader:
                model.train()
                mlp.train()
                opt.zero_grad()
                
                X = model(idx)
                logits = mlp(X)
                preds = torch.argmax(logits, dim=1)
                correct_train += torch.sum(preds == labels[idx]).float() 
                loss = loss_fn(logits, labels[idx])
                loss.backward()
                opt.step()
                t_loss += loss.item()
                del(X, logits, loss, preds)
                torch.cuda.empty_cache()
                gc.collect()
                
                model.eval()
                mlp.eval()

            with torch.no_grad():
                correct_test = 0 
                for idx in test_loader:
                    X = model(idx)
                    logits = mlp(X)
                    preds = torch.argmax(logits, dim=1)
                    correct_test = torch.sum(preds == labels[idx]).float()
            train_acc = correct_train/len(train_idx)*100
            test_acc = correct_test/len(test_idx)*100
            if test_acc > best_acc:
                best_acc = test_acc
            tq.set_description("Train acc = %.4f, Test acc = %.4f, Best acc = %.4f" % (train_acc.item(), test_acc.item(), best_acc))
            
if __name__ == '__main__':

    model = PointCloudFeatLearning(args.raw_dir, args.kernel_type, args.threshold, args.device).to(args.device)
    mlp = MLP(model.input_dim, args.hidden_dim, model.num_labels, args.num_layers).to(args.device)
    train(model, mlp)