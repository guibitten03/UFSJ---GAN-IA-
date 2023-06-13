import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()
        
        # Definir camadas lineares ocultas da rede
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        
        # Ultima camada linear que produzirá a saida binária
        self.fc6 = nn.Linear(hidden_dim, output_size)
        
        # Camada de Dropout
        self.dropout = nn.Dropout(0.3)
        
        
    def forward(self, x):
        # Achatar a imagem
        x = x.view(-1, 28*28)
        
        '''
        Por que usar Leaky Relu?
        
        Evita o problema do gradiente zero: A Leaky ReLU resolve o problema de gradientes zero em regiões negativas, que é uma limitação da 
        função ReLU padrão. A função Leaky ReLU introduz um pequeno vazamento (um valor negativo pequeno) para as entradas negativas, 
        permitindo que o gradiente flua mesmo quando a ativação é negativa.

        Encoraja a aprendizagem robusta: Ao introduzir um pequeno vazamento, a Leaky ReLU ajuda a evitar o desligamento de neurônios e a 
        aprendizagem excessivamente sensível a pequenas mudanças nas entradas. Isso pode melhorar a capacidade do modelo de generalizar e 
        produzir resultados mais robustos.

        Reduz o problema de saturação de gradientes: A função Leaky ReLU também pode ajudar a reduzir o problema de saturação de gradientes 
        em regiões positivas. A função ReLU padrão pode "matar" gradientes positivos quando a ativação é maior que zero, mas a Leaky ReLU
        permite que o gradiente flua livremente nessas regiões, evitando a saturação e melhorando a estabilidade do treinamento.

        Não introduz não linearidades excessivas: A Leaky ReLU é uma função de ativação não linear, mas possui um comportamento linear 
        para entradas positivas. Isso evita introduzir não linearidades excessivas que podem levar a problemas de explosão de gradiente 
        ou distorção dos resultados.
        '''
        
        x = F.leaky_relu(self.fc1(x), 0.2) # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc4(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc5(x), 0.2)
        x = self.dropout(x)
        
        out = self.fc6(x)

        return out
    
    
class Generator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()
        
        # Aqui ele recebe 784 caracteristicas - input_size
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc6 = nn.Linear(hidden_dim, output_size)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2) # (input, negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc4(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc5(x), 0.2)
        x = self.dropout(x)
        
        # Camada final com tanh aplicada
        '''
        Por que usar uma função TANH na ultima camada do gerador?
        
        Saída no intervalo [-1, 1]: A função tanh mapeia os valores para o intervalo entre -1 e 1. Isso é desejável em muitos casos, 
        especialmente quando se trabalha com dados normalizados ou quando as imagens geradas precisam ter valores de pixel no intervalo [-1, 1].

        Não linearidade: A função tanh é uma função não linear que introduz não linearidades nas saídas do gerador. Isso permite ao gerador 
        capturar relacionamentos mais complexos entre as entradas de ruído aleatório e as saídas geradas.

        Evita desvanecimento do gradiente: A função tanh não sofre do problema de desvanecimento do gradiente tão severamente quanto a função 
        sigmoid, especialmente nas regiões próximas aos extremos (-1 e 1). Isso ajuda a melhorar a estabilidade do treinamento da GAN, 
        permitindo que o gradiente flua mais livremente durante a retropropagação e evitando gradientes muito pequenos ou saturados.

        Simetria: A função tanh é simétrica em torno de zero, o que pode ser útil quando se trabalha com dados simétricos ou quando se deseja 
        que as saídas do gerador sejam simétricas em relação a algum ponto central.


        '''
        out = F.tanh(self.fc6(x))

        return out