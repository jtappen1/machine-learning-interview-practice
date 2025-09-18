# PyTorch Notes:

## Designing the Neural Net
### Forward Pass
- Forward(X (Input)) -> Does a forward pass of your data through the model.  What this does is takes the input data, brings it through the model, in the end returning logits.
- Logits -> The raw output of the neural network after the forward pass. Not yet probabilities, just normalized scores.
- Returns shape [batchsize, classes] for classification

### Initializing the NN:
- In our case, we needed to flatten the input tensor to be shape [batch_size, 784].  This is because the first linear layer was of size 784, 512.  They had to match.  To do this we used flatten().
- nn.flatten() -> when called with the default constructor flattens the dimensions from 1 -> n. It takes a start and end though to flatten specific dimensions.
- nn.Linear:  A Linear Layer: nn.Linear(in_features, out_features): represents the transformation $y= xW^T + b$
    - x: input tensor of shape [batch_size, in_features]
    - W: weight matrix of shape [out_features, in_features]
    - b: bias vector of shape [out_features]
    - y: output tensor of shape [batch_size, out_features]


### Activation Functions:
Activation functions add Non-linearity.  Multiple Linear layers is essentially the same as having a single one, on their own they just do transforms on the data. No matter how many linear layers we add, the network would only be able to represent linear functions.

- ReLU: $ReLU(x)=max(0,x)$ : -> If the input is positive, it passes through unchanged, otherwise it becomes zero.
    - Helps avoid the issue of vanishing gradients for positive inputs, as well as it is very fast.

### Loss Functions:


### Useful Things to Know:
- Vanishing Gradients: During backprop, if the derivative of an activation is very small, <1, repeated multiplication through layers can make gradients tiny -> weights stop updating

### Autograd:
- requires_grad : -> Tells a tensor that it needs to be able to compute the gradients of a loss ufnction with respect to some variables.
- with torch.no_grad(): -> we only want to do forward computations through the network. (.detatch() also works.)
