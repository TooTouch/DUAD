import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, flatten_features: int = 4096, latent_dim: int = 10):
        super(AutoEncoder, self).__init__()

        
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,      # input height
                out_channels=16,    # n_filters
                kernel_size=3,      # filter size
                stride=1,           # filter movement/step
                padding=1,      
            ),      
            nn.LeakyReLU(),    # activation
            nn.Conv2d(
                in_channels=16,      # input height
                out_channels=32,    # n_filters
                kernel_size=3,      # filter size
                stride=1,           # filter movement/step
                padding=1,      
            ),     
            nn.LeakyReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),     
            nn.Conv2d(
                in_channels=32,      # input height
                out_channels=32,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/stepP
                padding=2,      
            ),      
            nn.LeakyReLU(),    # activation
            nn.Conv2d(
                in_channels=32,      # input height
                out_channels=64,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      
            ),      
            nn.LeakyReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),   
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,      # input height
                out_channels=32,    # n_filters
                kernel_size=2,      # filter size
                stride=2,           # filter movement/step
                padding=0,      
            ),     
            nn.LeakyReLU(),       # activation 
            nn.Conv2d(
                in_channels=32,      # input height
                out_channels=32,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      
            ),      
            nn.LeakyReLU(),    # activation
           nn.ConvTranspose2d(
                in_channels=32,      # input height
                out_channels=16,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      
            ),
            nn.Conv2d(
                in_channels=16,      # input height
                out_channels=16,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,     
            ),      
            nn.LeakyReLU(),    # activation       
             nn.ConvTranspose2d(
                in_channels=16,      # input height
                out_channels=16,    # n_filters
                kernel_size=2,      # filter size
                stride=2,           # filter movement/step
                padding=0,      
            ), 
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=16,      # input height
                out_channels=16,    # n_filters
                kernel_size=3,      # filter size
                stride=1,           # filter movement/step
                padding=1,      
            ),     
            nn.LeakyReLU(),    # activation            
           nn.ConvTranspose2d(
                in_channels=16,      # input height
                out_channels=3,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      
            ),
            nn.Conv2d(
                in_channels=3,      # input height
                out_channels=in_channels,    # n_filters
                kernel_size=3,      # filter size
                stride=1,           # filter movement/step
                padding=1,      
            ),    
            nn.ReLU(), # activation 
        )        

        self.enc_linear = nn.Linear(flatten_features, latent_dim)
        self.dec_linear = nn.Linear(latent_dim, flatten_features)

    def encode(self, x):
        x_out = self.encoder(x) # B x 64 x 8 x 8
        x_size = x_out.size()

        z = self.enc_linear(x_out.flatten(start_dim=1))

        return z, x_size

    def decode(self, z, x_size):
        z = self.dec_linear(z)
        x = z.view(x_size)

        out = self.decoder(x)

        return out