import warnings
warnings.filterwarnings("ignore")
from COVID_data.read_knn import read_data,read_data_persistence, get_dataloaders
from models.gnn import GCN, GIN, GAT, SAGE, GraphTransformerWithPE
from argparse import ArgumentParser
from torch_geometric.transforms import AddRandomWalkPE
import torch
from torchinfo import summary
from tqdm import tqdm
import numpy as np

parser = ArgumentParser(description="KNN GNN")
parser.add_argument('--raw_dir', type=str, default = 'COVID_data', help="Directory where the raw data is stored")
parser.add_argument('--full', action='store_true', help="Directory where the raw data is stored")
parser.add_argument('--task', type=str, default = 'treatment', help="Task on PDO data")
parser.add_argument('--model', type=str, default = 'GCN', help="Directory where the raw data is stored")
parser.add_argument('--hidden_dim', type=int, default= 160, help="Hidden dim for the MLP")
parser.add_argument('--num_layers', type=int, default= 3, help="Number of MLP layers")
parser.add_argument('--batch_size', type=int, default= 16, help="Batch size")
parser.add_argument('--num_neighbors', type=int, default= 5, help="Number of neighbors for KNN graph")
parser.add_argument('--lr', type=float, default= 3e-3, help="Learnign Rate")
parser.add_argument('--wd', type=float, default= 3e-4, help="Weight decay")
parser.add_argument('--num_epochs', type=int, default= 10, help="Number of epochs")
parser.add_argument('--gpu', type=int, default= 0, help="GPU index")

def test(loader):
    model.eval()
    correct = 0
    # mse = 0
    for data in loader:  
        out = model(data.x, data.edge_index, data.batch)  
        preds = out.argmax(dim=1)
        correct += torch.sum(preds == data.y.long()).float()
        # mse += (torch.nn.functional.mse_loss(out, data.y) * len(data.y))
    return correct*100 / len(loader.dataset)  


def train(model, train_loader, test_loader):
    opt = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = args.wd)
    loss_fn = torch.nn.CrossEntropyLoss()
    best_acc = 0
    with tqdm(range(args.num_epochs)) as tq:
        for epoch in enumerate(tq):
        
            model.train()
            
            for step, data in enumerate(train_loader):  
                out = model(data.x, data.edge_index, data.batch)
                # loss = loss_fn(out, data.y)*1000
                loss = loss_fn(out, data.y.long())
                loss.backward()
                opt.step()
                opt.zero_grad()
            
            train_acc = test(train_loader)
            test_acc = test(test_loader)
            if test_acc > best_acc:
                best_acc = test_acc 
            tq.set_description("Train acc = %.4f, Test acc = %.4f, Best acc = %.4f" % (train_acc.item(), test_acc.item(), best_acc))
    return best_acc

args = parser.parse_args()
if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

if __name__ == '__main__':
    print(args)
    graphs, num_labels, train_idx, test_idx = read_data(args.raw_dir, args.num_neighbors, args.full)
    transform = AddRandomWalkPE(walk_length = graphs[0].x.shape[1], attr_name = 'pe')
    if(args.raw_dir == "pdo_data"):
        for i in range(len(graphs)):
            graphs[i].pe = torch.zeros_like(graphs[i].x)
    else:
        graphs = [transform(graph) for graph in tqdm(graphs)]
    
    train_loader, test_loader = get_dataloaders(graphs, train_idx, test_idx)
    mse = []
    for i in range(5):
        # model = GraphTransformerWithPE(graphs[0].x.shape[1], args.hidden_dim, num_labels, args.num_layers, 4, graphs[0].x.shape[1])
        if args.model == 'GCN':
            model = GCN(graphs[0].x.shape[1], args.hidden_dim, num_labels, args.num_layers).float()
        elif args.model == 'GIN':
            model = GIN(graphs[0].x.shape[1], args.hidden_dim, num_labels, args.num_layers).float()
        elif args.model == 'GAT':
            model = GAT(graphs[0].x.shape[1], args.hidden_dim, num_labels, args.num_layers).float()
        elif args.model == 'SAGE':
            model = SAGE(graphs[0].x.shape[1], args.hidden_dim, num_labels, args.num_layers).float()
        mse.append(train(model, train_loader, test_loader).item())
    mse = np.array(mse)
    print(f"Average:{mse.mean()}, Std:{mse.std()}")