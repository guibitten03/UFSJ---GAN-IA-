# import all libraries
import torch
from torch import nn

'''
            Por que usar camadas convolucionais transpostas?
            
            Recuperação de detalhes: A convolução transposta permite aumentar a resolução espacial de um mapa de recursos. 
            Ao aplicar a convolução transposta em uma imagem ou mapa de recursos de baixa resolução, é possível recuperar detalhes 
            finos que foram perdidos durante o downsampling ou a compressão.

            Geração de imagens de maior resolução: A convolução transposta é capaz de gerar imagens de maior resolução em comparação 
            com outras técnicas de upsampling, como a interpolação bilinear. Ao usar a convolução transposta, é possível gerar imagens 
            de alta qualidade com maior nitidez e detalhes.

            Aprendizado de características espaciais: A convolução transposta permite que a rede neural aprenda características espaciais
            importantes durante o treinamento. Ao gerar imagens ou realizar upsampling usando a convolução transposta, a rede neural é capaz 
            de aprender padrões e estruturas espaciais complexas que são essenciais para tarefas como geração de imagens realistas.

            Conectividade local: A convolução transposta preserva a conectividade local entre os pixels, permitindo que a rede neural 
            capture relações espaciais e contextuais dentro da imagem. Isso é especialmente importante para a geração de imagens coerentes 
            e realistas, onde os pixels vizinhos devem estar relacionados para manter a continuidade e a consistência visual.
'''

'''
            Estabilização do treinamento: O Batch Normalization ajuda a estabilizar o treinamento da rede neural, especialmente em redes mais
            profundas. Ele normaliza as ativações de cada camada, o que ajuda a mitigar problemas como o desvanecimento do gradiente e a 
            explosão do gradiente. Isso resulta em um treinamento mais estável e uma convergência mais rápida.

            Regularização: O Batch Normalization age como uma forma de regularização, reduzindo o overfitting. Ele introduz um pouco 
            de ruído nos dados normalizados de cada minibatch, o que ajuda a evitar o ajuste excessivo aos dados de treinamento. 
            Isso permite que a rede neural generalize melhor para dados de teste.

            Redução da dependência da inicialização dos pesos: O Batch Normalization reduz a dependência da inicialização cuidadosa dos
            pesos da rede. Ele torna a rede neural menos sensível à escolha inicial dos pesos, o que facilita o processo de inicialização 
            e ajuste dos hiperparâmetros.

            Redução da sensibilidade a mudanças de escala e translação: O Batch Normalization torna a rede neural menos sensível a mudanças 
            de escala e translação nos dados de entrada. Ele normaliza as ativações dentro de cada minibatch, o que ajuda a tornar a rede 
            mais robusta a variações nos dados de entrada.

            Aceleração do treinamento: O Batch Normalization pode acelerar o treinamento da rede neural, permitindo a utilização de taxas 
            de aprendizado mais altas. Como as ativações são normalizadas, problemas como grandes valores de gradiente são mitigados, o que 
            permite a utilização de taxas de aprendizado mais agressivas.
'''

class Generator(nn.Module):
    def __init__(self, noise_channels, image_channels, features):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            
            # Transpose block 1
            nn.ConvTranspose2d(noise_channels, features*16, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),

            # Transpose block 2
            nn.ConvTranspose2d(features*16, features*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*8),
            nn.ReLU(),

            # Transpose block 3
            nn.ConvTranspose2d(features*8, features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*4),
            nn.ReLU(),

            # Transpose block 4
            nn.ConvTranspose2d(features*4, features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*2),
            nn.ReLU(),

            # Last transpose block (different)
            nn.ConvTranspose2d(features*2, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, image_channels, features):
        super(Discriminator, self).__init__()
    
        # define the model
        self.model = nn.Sequential(
            # define the first Conv block
            nn.Conv2d(image_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # Conv block 2 
            nn.Conv2d(features, features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2),
    
            # Conv block 3
            nn.Conv2d(features*2, features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2),

            # Conv block 4
            nn.Conv2d(features*4, features*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2),

            # Conv block 5 (different)
            nn.Conv2d(features*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)