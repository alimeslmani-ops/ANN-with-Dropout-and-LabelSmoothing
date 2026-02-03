# Neural Network Training with Inverted Dropout & Label Smoothing

## 1. Description / Overview

A complete algorithm that performs feedforward with **Inverted Dropout** & **Label Smoothing**,
then performs backpropagation to compute all the gradients and update the parameters
(**weights & biases**) from scratch only using numpy.

The functions of the algorithm are implemented manually, meaning the layers initialization,
feedforward, and backpropagation are implemented **layer by layer**.

This algorithm is a simple demonstration of ANN training. For that reason, the network was
trained on randomly initialized data with a simple network architecture that consists of:

- **2 inputs**
- **3 hidden neurons**
- **2 outputs**

The output labels are **one-hot encoded** to suit the label smoothing algorithm.

The training phase of the random dataset is done using **mini-batches**, which are small subsets
of training data used for a single gradient update. The dataset is shuffled and then yields
mini-batches of:

```
(X_batch , Y_batch)
```

---

## 2. Inverted Dropout & Label Smoothing

It is better for network training to include some form of **regularization** to reduce
overfitting and improve generalization.

### Inverted Dropout

In this algorithm, **Inverted Dropout** is implemented for the hidden layers. It randomly
disables neurons during training, effectively training a different sub-network at each iteration
and preventing co-adaptation of neurons.

### Mathematical Formulation

```
h̃^(L) = (r_i^(L) / p) ⨀ h^(L)
```

Where:

- `p` → keep probability  
- `r_i^(L)` → Bernoulli mask  
- `r_i = 1` → neuron is kept  
- `r_i = 0` → neuron is dropped  

During the feedforward phase, some neurons are disabled based on the Bernoulli mask.
During backpropagation, if a neuron is dropped:

- activation = 0  
- gradient = 0  

Therefore, **no weight update occurs** for that neuron.

---

### Label Smoothing

Label smoothing is implemented for the **output layer** and is optional depending on the output
layer type. Since label smoothing applies to **one-hot encoded labels**, it can be enabled
inside the `feed_forward()` function by setting the `smoothing` parameter to `True`.

Label smoothing replaces hard one-hot labels with soft labels, preventing overconfident
predictions and regularizing the output layer.

### Mathematical Formulation

```
y_k^LS = (1 − α) y_k + α / k
```

Where:

- `y_k` → original one-hot label  
- `α` → smoothing parameter  
- `k` → number of classes  

> **Note:**  
> This algorithm does not use **L1** or **L2** regularization, but they may be added in the future.

---

## 3. Activations & Derivatives

The following activation functions are implemented:

- **Sigmoid** → used in the hidden layer during feedforward  
- **Softmax** → used in the output layer since the labels are one-hot encoded  

The derivatives of these activation functions are also implemented for the backpropagation phase.

The function `derivative_softmax()` is included **only for demonstration** and is not used in
practice, because computing the full Jacobian is complex.

Luckily, a simplified derivative can be used when combining **Softmax + Cross-Entropy Loss**:

```
d/dx = a − y
```

This simplification works because the derivative of the **softmax activation function** and the
**cross-entropy loss function** together reduces to `a − y`.

Loss functions are implemented for demonstration purposes only. However, it is essential to
know which loss function is used in order to compute its derivative during backpropagation.

In this algorithm, the loss function used is **Cross-Entropy Loss**.

---

## 4. Network Architecture

The network architecture is:

```
2 → 3 → 2
```

- 2 input neurons  
- 3 hidden neurons  
- 2 output neurons  

The function `create_ANN()` initializes the ANN with the following parameters:

- **W1**: weights from input to hidden layer  
  - Shape: `(3 , 2)`  

- **b1**: bias for hidden layer  
  - Shape: `(3 , 1)`  

- **W2**: weights from hidden to output layer  
  - Shape: `(2 , 3)`  

- **b2**: bias for output layer  
  - Shape: `(2 , 1)`  

### Feedforward Data Flow

In the `feed_forward()` function, the data flow is carefully designed to match the network
architecture.

- Input batch size: `m`  
- Input shape: `(m , 2)`  
- Transposed input: `(2 , m)`  

Matrix multiplication:

```
z1 = W1 · X_T
```

Shape validation:

```
(3 , 2) · (2 , m) → Valid
```

This produces the activation matrix `a1` with shape:

```
(3 , m)
```

The same process is repeated during both **feedforward** and **backpropagation** phases.

---

## 5. Usage

Currently, this algorithm is intended for **educational and demonstration purposes only** and
does not have a specific real-world application.

*(Maybe later though ☺)*
