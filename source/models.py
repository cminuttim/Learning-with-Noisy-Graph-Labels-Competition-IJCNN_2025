# source/models.py - Contains all model definitions
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool


# Edge-aware encoder with classification capability
class EdgeEncoder(MessagePassing):
    def __init__(self, in_channels, edge_dim, hidden_dim):
        super(EdgeEncoder, self).__init__(aggr='add')  # Message aggregation
        self.node_mlp = torch.nn.Linear(in_channels + hidden_dim, hidden_dim)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, hidden_dim),
            torch.nn.LeakyReLU(0.15),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        edge_emb = self.edge_mlp(edge_attr)  # Transform edge features
        return self.propagate(edge_index, x=x, edge_attr=edge_emb)

    def message(self, x_i, x_j, edge_attr):
        z = torch.cat([x_i, edge_attr], dim=1)  # Concatenate node and edge features
        return self.node_mlp(z)

    def update(self, aggr_out):
        return aggr_out
    
class EdgeVGAE(torch.nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, latent_dim, num_classes):
        super(EdgeVGAE, self).__init__()
        self.encoder = EdgeVGAEEncoder(input_dim, edge_dim, hidden_dim, latent_dim)
        self.classifier = torch.nn.Linear(latent_dim, num_classes)  # Classifier head
        
        # MLP for edge attribute reconstruction
        #self.edge_mlp = torch.nn.Linear(latent_dim * 2, edge_dim)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(latent_dim * 2, latent_dim),
            torch.nn.LeakyReLU(0.15),
            torch.nn.Linear(latent_dim, edge_dim)
        )

        # Initialize weights using Kaiming initialization
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                # Apply Kaiming initialization for LeakyReLU
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.15)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
                    
    def forward(self, x, edge_index, edge_attr, batch, eps=None):
        mu, logvar = self.encoder(x, edge_index, edge_attr)
        if eps==0.0:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)  # Sample latent variable

        class_logits = self.classifier(global_mean_pool(z, batch))  # Graph-level classification
        return z, mu, logvar, class_logits

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)  # Clamp values to prevent extreme exponentiation
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std



    def decode(self, z, edge_index):
        # Predict adjacency matrix
        adj_pred = torch.sigmoid(torch.mm(z, z.t()))

        # Predict edge attributes using an MLP
        row, col = edge_index
        edge_features = torch.cat([z[row], z[col]], dim=-1)  # Concatenate node embeddings
        edge_attr_pred = self.edge_mlp(edge_features)
        edge_attr_pred = torch.sigmoid(edge_attr_pred)  # Assuming attributes are in [0,1]
        
        return adj_pred, edge_attr_pred
    
    def recon_loss(self, z, edge_index, edge_attr):
        adj_pred, edge_attr_pred = self.decode(z, edge_index)

        # Build adjacency ground truth
        adj_true = torch.zeros_like(adj_pred, dtype=torch.float32)
        adj_true[edge_index[0], edge_index[1]] = 1.0  

        # Loss for adjacency matrix reconstruction (BCE Loss)
        adj_loss = F.binary_cross_entropy(adj_pred, adj_true)

        # Loss for edge attribute reconstruction (MSE Loss)
        edge_attr_pred_selected = edge_attr_pred  
        edge_loss = F.mse_loss(edge_attr_pred_selected, edge_attr)
        #return adj_loss
        return 0.1*adj_loss + edge_loss



    def kl_loss(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-10, max=10)  # Prevent extreme values
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

class EdgeVGAEEncoder(torch.nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, latent_dim):
        super(EdgeVGAEEncoder, self).__init__()
        self.conv1 = EdgeEncoder(input_dim, edge_dim, hidden_dim)
        self.conv2 = EdgeEncoder(hidden_dim, edge_dim, hidden_dim)
        self.drop = torch.nn.Dropout(0.05)

        # Mean and log variance layers
        self.mu_layer = torch.nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = torch.nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.drop(x)
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr), 0.15)
        x = self.drop(x)
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr), 0.15)
        # x = self.drop(x)
        return self.mu_layer(x), self.logvar_layer(x)  # Return mean and log variance


