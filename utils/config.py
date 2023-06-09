class Configuration():
    def __init__(self, 
                 input_size:int=784,
                 d_output_size:int=1,
                 d_hidden_size:int=256,
                 z_size:int=64,
                 g_output_size:int=784,
                 g_hidden_size:int=256):
        
        # Discriminator hyperparams ---------------------

        # Size of input image to discriminator (28*28)
        self.input_size = input_size
        # Size of discriminator output (real or fake)
        self.d_output_size = d_output_size
        # Size of last hidden layer in the discriminator
        self.d_hidden_size = d_hidden_size

        # Generator hyperparams -------------------------

        # Size of latent vector to give to generator
        self.z_size = z_size
        # Size of discriminator output (generated image)
        self.g_output_size = g_output_size
        # Size of first hidden layer in the generator
        self.g_hidden_size = g_hidden_size