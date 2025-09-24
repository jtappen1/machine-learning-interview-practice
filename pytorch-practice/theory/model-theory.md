# ML Theory:

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
Formula $$ BCE = - [y \cdot log(\hat{y}) + (1-y)\cdot log(1-\hat{y})]$$


## Activation Functions:
The point of Activation functions is to add non-linearity into systems that would be otherwise linear. This helps the models learn to predict other patterns that would not be possible with a simple linear model. Also, these are the gradients, not the function themselves values.

ReLU Family: 
Used generally in hidden layers.

- ReLU: $f(x) = max(0,x)$ essentially sets a lower threshold at zero. Anything below zero is zero.  It is very cheap to do and prevents the vanishing gradient problem.
- Leaky ReLU: Allows a small non-zero gradient $\alpha$ for negative inputs.
    - This solves the Dying ReLU problem.  If a neurons' input is consistently negative, it will output a zero gradient due to Relu.  This helps Neurons form becoming permantly inactive


## Optimizers:
SGD:
- Computes the gradient using only a single randomly chosen data point. The weights are then updated for every single data point.  The Stoastic part comes from the randomness in which datapoint you pick.
- Momentum is the main hyperparameter.  
- Overall: Good baseline, very sensitive to noise but due to that sensitivity, can get out of local minimas.

Adam: 
- Combines Adagrad and RMSprop, incorporates momentum as well as an exponential decaying average of past squared gradients to scale the learning rate.
- Has quick convergence, good for complex models.

## Scoring
Precision:
- Precision measures the proportion of relevant retrieved instances compared to all of the elements that were retrieved.
- $$Precision = \frac{TP}{TP + FP}$$
- Essentially, how accurate were we when predicting a person was a person.  How precise my predictions are.

Recall:
- Recall measures the amount of relevant retrieved instances compared to all relevant instances.
- $$Recall = \frac{TP}{TP + FN}$$
- Essentially, how accurate were we on all real occurances. Tells us how may of the True Positive cases the Model actually found.
- Out of all the actual cats that exist, how many did I successfully remember and identify?

F1 Score: 
- Essentially the Harmonic mean of precision and recall.  High if both are high.
- $$ F1 = 2 \cdot \frac{Precision * Recall}{Precision + Recall}$$