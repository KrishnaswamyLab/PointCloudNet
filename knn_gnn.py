from melanoma_data.read_data import read_data, get_dataloaders
from models.gnn import GCN, GIN, GAT, SAGE
from argparse import ArgumentParser
import torch
from tqdm import tqdm

parser = ArgumentParser(description="KNN GNN")
parser.add_argument('--raw_dir', type=str, default = 'data/melanoma/raw', help="Directory where the raw data is stored")
parser.add_argument('--model', type=str, default = 'SAGE', help="Directory where the raw data is stored")
parser.add_argument('--hidden_dim', type=int, default= 20, help="Hidden dim for the MLP")
parser.add_argument('--num_layers', type=int, default= 3, help="Number of MLP layers")
parser.add_argument('--batch_size', type=int, default= 64, help="Batch size")
parser.add_argument('--num_neighbors', type=int, default= 5, help="Number of neighbors for KNN graph")
parser.add_argument('--lr', type=float, default= 1e-2, help="Learnign Rate")
parser.add_argument('--wd', type=float, default= 3e-4, help="Weight decay")
parser.add_argument('--num_epochs', type=int, default= 100, help="Number of epochs")
parser.add_argument('--gpu', type=int, default= 0, help="GPU index")

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  
        out = model(data.x, data.edge_index, data.batch)  
        preds = out.argmax(dim=1)
        correct += torch.sum(preds == torch.LongTensor(data.y)).float()
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
                loss = loss_fn(out, torch.LongTensor(data.y))
                loss.backward()
                opt.step()
                opt.zero_grad()
            
            train_acc = test(train_loader)
            test_acc = test(test_loader)
            if test_acc > best_acc:
                best_acc = test_acc 
            tq.set_description("Train acc = %.4f, Test acc = %.4f, Best acc = %.4f" % (train_acc.item(), test_acc.item(), best_acc))

args = parser.parse_args()
if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

if __name__ == '__main__':
    graphs, num_labels, train_idx, test_idx = read_data(args.raw_dir, args.num_neighbors)
    train_loader, test_loader = get_dataloaders(graphs, train_idx, test_idx)
    if args.model == 'GCN':
        model = GCN(graphs[0].x.shape[1], args.hidden_dim, num_labels, args.num_layers)
    elif args.model == 'GIN':
        model = GIN(graphs[0].x.shape[1], args.hidden_dim, num_labels, args.num_layers)
    elif args.model == 'GAT':
        model = GAT(graphs[0].x.shape[1], args.hidden_dim, num_labels, args.num_layers)
    elif args.model == 'SAGE':
        model = SAGE(graphs[0].x.shape[1], args.hidden_dim, num_labels, args.num_layers)
    train(model, train_loader, test_loader)