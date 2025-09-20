# Model Theory:

## CNN:
- conv2D Layers: (input channels, output channels, kernel size, stride)
-   produces a feature map, or an activation map.  Each "feature" is the learned value of the dot product between the kernel and the (nxn) batch that matches up with the kernel.
- feature map


## Loss Functions:
Norms: Fuction that measures the size/length of a vector, relative to the origin (zero vector).  Must pass the the 3 rules.  L2 norm is essentially the Euclidian Distance.  L1 norm is essentially the Manhattan distance, or x = (3, 4) l1 norm = 7 while l2 norm = 5

Error is simply the difference between $\hat{y}$ and y,  so $E = y - \hat{y} $


Regression Losses:
- MSE Loss (nn.MSELoss()) L2 Loss: Calculates the average of the squared errors. It heavily penalizes large errors, forcing the model to be very precise. Use: Standard choice for position and orientation estimation.
    - Called L2 Loss becuase it's mathematical formulation is derived directly from the L2 Norm (Also known as the Euclidian Distance)
- MAE Loss (nn.L1Loss()) L1 Loss: Calculates the average of the absolute errors. It's more robust to outliers (rare, incorrect sensor readings) because it doesn't square the large errors. Use: Can be better than MSE for very noisy sensor data.

Classification Losses:
- Cross Entropy Loss (nn.CrossEntropyLoss()): Used for multi-class classification, classifying a point as X, Y, or Z.  Designed to penalize confident wrong predictions heavily
- Cross Entropy measures the difference between two probability distrubutions, the True Distribution (GT Label), and Predicted Dist (Output of our model). 
- Quantifies how well the Predicted Probability matches the one-hot encoded true label.
- Cross-Entropy Loss heavily penalizes the model when it assigns a low probability to the correct class (the "Bad Prediction" scenario)
Formula : $$ CE = - \sum{i=1}{C} y_{i} \cdot log(\hat{y_{i}})$$
- This works because of the one hot vector.  All wrong probabilites are 0.  However, the correct class multiplied by the predicted probability will be the only thing that impacts the loss. There are a few situations: 
    - First, we remember that what the softmax activation function does is put the probability of the logits between 0 and 1, where they all sum up to 1.
    -  Say that the correct class is in index 2, but our probabilities are high on  0.  That means the corresponding probabilities for 1 and 2 are going to be low. The log of a small number is large. Therefore the -log(prob) of the correct class will be low, which will equal a large loss value.  We want to minimize the loss, so it will heavily penalize confident incorrect choices.

- Binary Cross Entropy (nn.BCELoss()) : Used for binary classifcation:  More stable and includes the Sigmoid activation
- (nn.BCEWithLogitsLoss()): This is applied directly to the logits and is the preferered method of using BCE Loss.  This is because it uses the trick to handle very small/large log-probabilities.  Either way, BCE requires a sigmoid before to squash the probabilites between 0, 1.
- Binary Cross Extropy Loss essentially does the same thing.  However, it has only two choices.  A sigmoid is used to squash the probabilities between 0 and 1. 
Formula $$ BCE = - [y \cdot log(\hat{y}) + (1-y)\cdot (1-\hat{y})]$$

