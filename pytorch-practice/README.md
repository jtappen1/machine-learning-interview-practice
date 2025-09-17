# PyTorch Notes:

## Designing the Neural Net

### Forward Pass
- Forward(X (Input)) -> Does a forward pass of your data through the model.  What this does is takes the input data, brings it through the model, in the end returning logits.
- Logits -> The raw output of the neural network after the forward pass. Not yet probabilities, just normalized scores.
- Returns shape [batchsize, classes] for classification


