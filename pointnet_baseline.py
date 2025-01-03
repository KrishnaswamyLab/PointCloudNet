import warnings
warnings.filterwarnings("ignore")
from models.pointnet import PointNetLoading, Pointnet_plus
from models.gnn import GCN, GIN, GAT, SAGE
from argparse import ArgumentParser
import torch
from tqdm import tqdm
import numpy as np

parser = ArgumentParser(description="KNN GNN")
parser.add_argument('--raw_dir', type=str, default = 'COVID_data', help="Directory where the raw data is stored")
parser.add_argument('--full', action='store_true', help="Directory where the raw data is stored")
parser.add_argument('--task', type=str, default = 'treatment', help="Task on PDO data")
parser.add_argument('--model', type=str, default = 'GCN', help="Directory where the raw data is stored")
parser.add_argument('--hidden_dim', type=int, default= 150, help="Hidden dim for the MLP")
parser.add_argument('--num_layers', type=int, default= 3, help="Number of MLP layers")
parser.add_argument('--batch_size', type=int, default= 32, help="Batch size")
parser.add_argument('--num_neighbors', type=int, default= 5, help="Number of neighbors for KNN graph")
parser.add_argument('--lr', type=float, default= 1e-3, help="Learnign Rate")
parser.add_argument('--wd', type=float, default= 3e-4, help="Weight decay")
parser.add_argument('--num_epochs', type=int, default= 20, help="Number of epochs")
parser.add_argument('--gpu', type=int, default= 0, help="GPU index")

def train(model, epochs):
    best_acc = 0
    best_val_acc = 0
    
    opt = model.configure_optimizers()
    
    with tqdm(range(epochs)) as tq:
        for epoch in enumerate(tq):
            t_loss = 0
            for data in train_loader:
                opt.zero_grad()
                loss = model.training_step(data, data.batch)
                loss.backward()
                opt.step()
                t_loss+=loss.item()
            model.eval()
            with torch.no_grad():
                for data in val_loader:
                    model.validation_step(data, data.batch)
                val_acc = model.on_validation_epoch_end()
                for data in test_loader:
                    model.test_step(data, data.batch)
                test_acc = model.on_test_epoch_end()
                if(val_acc>= best_val_acc):
                    best_val_acc = val_acc
                    best_acc = test_acc
            tq.set_description("Train loss = %.4f, Val acc = %.4f, Best val acc = %.4f, Best acc = %.4f" % (t_loss, test_acc, best_val_acc, best_acc))
    return best_acc

args = parser.parse_args()
if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

if __name__ == '__main__':
    print(args)
    acc = []
    for i in range(10):
        train_loader, val_loader, test_loader, input_dim, num_classes = PointNetLoading(args.raw_dir, args.full, args.batch_size, args.device)
        model = Pointnet_plus(1, input_dim, num_classes, args.lr).to(args.device)
        model.train()
        acc.append(train(model, args.num_epochs))
    acc = np.array(acc)
    print(f"Average performance: {acc.mean()}, std: {acc.std()}")