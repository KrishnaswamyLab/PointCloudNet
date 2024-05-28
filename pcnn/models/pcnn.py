import torch
import torch.nn as nn
import torch_geometric
import pytorch_lightning as pl
from pcnn.models.layers import BaseLayer
from pcnn.utils import compute_sparse_diffusion_operator, get_scattering_indices
from pcnn.models.legs import scatter_moments, ScatterAttention
from pcnn.models.learnable_graph import GraphLearningLayer
import wandb
import matplotlib.pyplot as plt

class PCNN(pl.LightningModule):
    def __init__(self,num_layers, input_dim, hidden_dim, num_classes, lr, compute_P, K = None, pooling = None, scattering_aggregate = False, learnable_graph = False, **kwargs):
        
        super().__init__()

        self.save_hyperparameters()
        
        self.num_layers = num_layers

        self.lr = lr

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.scattering_aggregate = scattering_aggregate # If true, stacks the different scattering features together.
        
        if kwargs['layer']['filter_method'] == "extract_scattering":
            J = kwargs['graph_construct']['J']
            n_norms = len(kwargs['graph_construct']['norm_list'])
            if kwargs["scattering_n_pca"] is not None:
                num_scattering_feats = kwargs["scattering_n_pca"]
            else: 
                num_scattering_feats = int((0.5*((J)*(J)+(J))+1)*n_norms*input_dim)
                #num_scattering_feats = ((1 * n_norms) + int(0.5*((J+1)*(2+J)) * n_norms )) * input_dim
            self.bypass_pooling = True
        else:
            num_scattering_feats = 0
            self.bypass_pooling = False

        self.layers = nn.ModuleList([BaseLayer(output_dim = hidden_dim, input_dim = input_dim, K = K, num_scattering_feats = num_scattering_feats, **kwargs["layer"])])
        if num_layers > 1:
            for _ in range(num_layers-1):
                #if hidden_dim is None:
                previous_output_dim = self.layers[-1].output_dim
                #else:
                #previous_output_dim = hidden_dim
                self.layers = self.layers.append(BaseLayer(output_dim = hidden_dim, input_dim = previous_output_dim, K= K, num_scattering_feats = num_scattering_feats,  **kwargs["layer"]))

        #self.model = nn.Sequential(*self.layers)

        if self.scattering_aggregate:
            output_dim = int((0.5*((hidden_dim)*(hidden_dim)+(hidden_dim))+1)*input_dim) 
        else:
            output_dim = self.layers[-1].output_dim

        if pooling is None: #by default, pooling is global mean pooling
            self.pooling = lambda graph : torch_geometric.nn.global_mean_pool(graph.x, graph.batch)
        elif pooling.name == "moments":
            self.pooling = lambda batch : scatter_moments( batch.x, batch.batch, pooling.moments_order)
            output_dim = output_dim * len(pooling.moments_order)
        elif pooling.name == "attention":
            self.pooling = ScatterAttention(in_channels= output_dim)


        self.classifier = nn.Sequential(nn.Linear( output_dim, int(output_dim/2)), nn.ReLU(), nn.Linear(int(output_dim/2),num_classes))

        self.loss_fun = torch.nn.CrossEntropyLoss()

        self.num_classes = num_classes

        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.compute_P = compute_P # wether to compute the diffusion operator in the forward pass
        self.K = K #number of diffusion steps

        self.learnable_graph = learnable_graph
        if self.learnable_graph:
            self.graph_layer = GraphLearningLayer()


    def forward(self, x):
        
        if self.learnable_graph:
            x = self.graph_layer(x)

        if self.compute_P: #Compute the powers of the diffusion operator
            P = compute_sparse_diffusion_operator(x)
            Pk_list = []
            Pk = P
            for k in range(self.K):
                Pk_list.append(Pk)
                Pk = torch.sparse.mm(Pk, P)
            x.Pk = Pk_list

        results = [x.clone()]
        for layer in self.layers:
            x = layer(x)
            results.append(x.clone())

        if self.scattering_aggregate:
            idx2 = get_scattering_indices(self.hidden_dim)
            x2 = results[2].x.reshape(results[2].x.shape[0],self.input_dim,self.hidden_dim,self.hidden_dim)
            x2 = torch.cat([x2[:,:,i,j] for i,j in idx2],-1)
            x_ = torch.cat((results[0].x,results[1].x,x2),dim=1)
            x.x = x_
        
        return x
    
    def predict_step(self,batch,batch_idx):
        y = batch.y
        graph_out = self(batch)
        if not self.bypass_pooling:
            pooled = self.pooling(graph_out)
        else:
            pooled = graph_out.x.float()

        y_hat = self.classifier(pooled)

        return {'y_hat': y_hat, 'batch':batch}


    
    def training_step(self, batch, batch_idx):
        y = batch.y
        graph_out = self(batch)
        if not self.bypass_pooling:
            pooled = self.pooling(graph_out)
        else:
            pooled = graph_out.x.float()

        y_hat = self.classifier(pooled)

        loss = self.loss_fun(y_hat,y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        y = batch.y
        graph_out = self(batch)

        if not self.bypass_pooling:
            pooled = self.pooling(graph_out)
        else:
            pooled = graph_out.x.float()
        y_hat = self.classifier(pooled)

        loss = self.loss_fun(y_hat,y)
        self.log('val_loss', loss)
            
        self.validation_step_outputs.append({'val_loss': loss,'y_hat': y_hat, 'y': y})
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        y_hat = torch.cat([x['y_hat'] for x in outputs]).cpu()
        y = torch.cat([x['y'] for x in outputs]).cpu()
        acc = torch.sum(y_hat.argmax(dim=1) == y).item() / (len(y) * 1.0)
        self.log('val_acc', acc)
        self.validation_step_outputs.clear()

        if self.learnable_graph:
            self.logger.log_table(key="Wavelet Scales Coeffs", data=[self.layers[-1].filtering_layer.wavelet_constructor.cpu().tolist()], columns = list(range(4)))
            self.log('epsilon', self.graph_layer.epsilon.cpu()) #loging the epsilon parameter of the graph learning layer

            plt.figure()
            plt.imshow(self.layers[-1].filtering_layer.wavelet_constructor.cpu(), cmap='viridis', interpolation='nearest')
            plt.colorbar()
            plt.title('Validation Heatmap')
            plt.xlabel('Exponent')
            plt.ylabel('Scales')
            wandb.log({'heatmap': plt})
            plt.show()

    
    def test_step(self, batch, batch_idx):
        y = batch.y
        graph_out = self(batch)
        if not self.bypass_pooling:
            pooled = self.pooling(graph_out)
        else:
            pooled = graph_out.x.float()
        y_hat = self.classifier(pooled)

        loss = self.loss_fun(y_hat,y)
        self.log('test_loss', loss)
        
        self.test_step_outputs.append({'test_loss': loss,'y_hat': y_hat, 'y': y})
        return loss
    
    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        y = torch.cat([x['y'] for x in outputs])
        acc = torch.sum(y_hat.argmax(dim=1) == y).item() / (len(y) * 1.0)
        self.log('test_acc', acc)
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            print(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()