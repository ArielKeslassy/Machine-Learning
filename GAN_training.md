# GAN training
---
## Exploring training methods for generative adversarial networks.
### General setting:
- Task: generating hand-written digits from [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
- Models:
  - Generator:
    - input: noise vectors of size (batch_size, 100)
    - output: images of size (batch_size, 1, 28, 28)
    - architecture: alternating linear and activation layers  
  - Discriminator:
    - input: images of size (batch_size, 1, 28, 28)
    - output: prediction vectors of size (batch_size, 1)
    - architecture: alternating linear and activation layers
  - Hyperparameters:
    - gen_layers: number of linear layers in the generator
    - dis_layers: number of linear layers in the discriminator
    - activation: type of activation function, one in [relu, leaky_relu, elu, gelu, relu6]
    - gen_dropout: use dropout in the generator
    - dis_dropout: use dropout in the discriminator

### Experiments:
- Basic experiment for comparison
  - configuration: gen_layers = 3, dis_layers = 2, activation = ReLU, gen_dropout = False, dis_dropout = False
  - results:
    - loss
    - ![samples (bottom row in real data for comparison)]()
- Activation functions: 
