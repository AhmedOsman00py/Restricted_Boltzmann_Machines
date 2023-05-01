# Graphical Model Project (M2 Data Science 2022/2023)

The Restricted Boltzmann Machine (RBM) is a type of neural network that was popularized in the early 2000s by Geoffrey Hinton and his colleagues.

RBM is a type of energy-based model that learns to represent the underlying probability distribution of input data by minimizing the energy function of the system. It is a two-layer neural network that consists of a visible layer and a hidden layer, where each neuron in the visible layer is connected to every neuron in the hidden layer but there are no connections between neurons within each layer.

The RBM is used in various applications, such as dimensionality reduction, feature learning, collaborative filtering, data augmentation and recommendation systems. It can also be used as a building block in more complex deep learning architectures, such as Deep Belief Networks (DBNs) and Convolutional Neural Networks (CNNs).

- Our work was done using PyTorch and is accessible in the `rbm.py` file.
- More details about RMBs are available in the notebook `mnist.ipynb`.

## Model checkpoints

Pretrained models are available in the `/save/model` folder and can be loaded as follows :

```python
# --- RBM
rbm = RBM(k=1)
rbm.load_state_dict(torch.load("/save/model/rbm.pth"))

# --- Classifiers with RBM features
clf = pickle.load(open("save/model/model_name.pkl", "rb"))
```

## References

* Geoffrey Hinton (2010) A Practical Guide to Training Restricted Boltzmann Machines, [UTML TR 2010–003](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)
* [https://github.com/odie2630463/Restricted-Boltzmann-Machines-in-pytorch](https://github.com/odie2630463/Restricted-Boltzmann-Machines-in-pytorch)

## License

Copyright © 2023 [Ahmed OSMAN](https://github.com/AhmedOsman00py). <br />
This project is [MIT License](https://github.com/AhmedOsman00py/Restricted_Boltzmann_Machines/blob/main/LICENSE) licensed.