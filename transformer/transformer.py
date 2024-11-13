import torch
import torch.nn as nn
import torch.nn.functional as F


class ParticleTransformer(nn.Module):
    def __init__(self, 
                 input_dim=5,           # Dimension of input (e.g. vel_x, vel_y, pos_x, pos_y, mass)
                 latent_dim=128,         # Latent space dimension D
                 hidden_dim=256,         # Hidden layer size in embedding network
                 embedding_layers=2,     # Number of layers in embedding network
                 transformer_layers=4,   # Number of transformer layers
                 n_heads=8,              # Number of attention heads
                 ff_dim=512,             # Feed-forward dimension in transformer
                 output_dim=5,           # Output dimension (same as input_dim in this case)
                 dropout=0.1             # Dropout rate
                ):
        super(ParticleTransformer, self).__init__()
        
        # Embedding Network (MLP) to convert input space to latent space
        mlp = []
        
        # First layer: from input_dim to hidden_dim
        mlp.append(nn.Linear(input_dim, hidden_dim))
        mlp.append(nn.ReLU())
        mlp.append(nn.Dropout(dropout))  # Optional regularization
        
        # Hidden layers: hidden_dim to hidden_dim
        for _ in range(embedding_layers - 2):
            mlp.append(nn.Linear(hidden_dim, hidden_dim))
            mlp.append(nn.ReLU())
            mlp.append(nn.Dropout(dropout))
        
        # Final layer: from hidden_dim to latent_dim
        mlp.append(nn.Linear(hidden_dim, latent_dim))
        mlp.append(nn.ReLU())
        mlp.append(nn.Dropout(dropout))

        # Sequential model for the embedding MLP network
        self.embedding = nn.Sequential(*mlp)
        
        # Transformer Encoder Layer with batch_first=True
        transformer_layer = nn.TransformerEncoderLayer(d_model=latent_dim, 
                                                       nhead=n_heads,
                                                       dim_feedforward=ff_dim,
                                                       dropout=dropout,
                                                       batch_first=True)  # Enable batch_first=True

        # Multiple stacked transformer layers
        self.transformer = nn.TransformerEncoder(transformer_layer, 
                                                 num_layers=transformer_layers)
        
        # Final projection layer to map back from latent_dim to output_dim
        self.decoder = nn.Linear(latent_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the model.
        :param x: Input tensor of shape (batch_size, n_particles, input_dim)
        """
        # Input shape: (batch_size, n_particles, input_dim)
        batch_size, n_particles, input_dim = x.size()
        
        # Embed input into latent space
        x = self.embedding(x.view(-1, input_dim))  # Reshape to (batch_size * n_particles, input_dim)
        x = x.view(batch_size, n_particles, -1)    # Reshape back to (batch_size, n_particles, latent_dim)
        
        # Since batch_first=True, the transformer expects input as (batch_size, n_particles, latent_dim)
        
        # Pass through transformer encoder (batch_first is now handled automatically)
        x = self.transformer(x)  # Output shape: (batch_size, n_particles, latent_dim)

        # Decode back to original particle dimensionality (e.g., velocities, positions, mass)
        x = self.decoder(x)  # Output shape: (batch_size, n_particles, output_dim)

        return x


# Test the module with configurable parameters
if __name__ == "__main__":
    # Example usage:
    input_dim = 5       # 5D particles (e.g., vel_x, vel_y, pos_x, pos_y, mass)
    latent_dim = 128    # Latent space dimension
    hidden_dim = 256    # Hidden units in embedding MLP
    output_dim = 5      # Output dimension (same as input)
    n_particles = 512   # Number of particles
    batch_size = 4      # Batch size for testing
    
    # Initialize ParticleTransformer
    model = ParticleTransformer(
        input_dim=input_dim,
        latent_dim=latent_dim, 
        hidden_dim=hidden_dim, 
        embedding_layers=3,       # 3 layers in embedding
        transformer_layers=4,     # 4 Transformer layers
        n_heads=8,                # 8 attention heads
        ff_dim=512,               # 512 in feed-forward layers
        output_dim=output_dim, 
        n_particles=n_particles, 
        dropout=0.1               # Dropout for regularization
    )
    
    # Create test input (random 4 batches of 512 particles, each with 5 features)
    sample_input = torch.rand(batch_size, n_particles, input_dim)
    
    # Forward pass through the model
    output = model(sample_input)
    
    print("Input shape:", sample_input.shape)    # Expected: (batch_size, n_particles, input_dim)
    print("Output shape:", output.shape)         # Expected: (batch_size, n_particles, output_dim)

    # Since you provided test code, make sure this runs without errors.
    output = model(sample_input)  # Ensure input is on the same device as the model if CUDA is used.
    print("Forward pass output shape:", output.shape)