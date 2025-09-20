# Constructing a Pipeline from Scratch
https://github.com/Devinterview-io/pytorch-interview-questions

# Steps:
1. Initialize the training and test dataset. Add necessary transforms and set up the dataloaders
    - Do necessary operations to download the data
    - Initialize the different transforms that will be done on the data. Do Transforms.Compose([]).  Some possible transforms ... 
        - ToTensor(): - > turns it into a tensor of CxHxW, and normalizes the values between 0 and 1.
        - transforms.Normalize(mean, std)
        - transforms.Resize(size)
        - transforms.RandomHorizontalFlip(prob)
        - transforms.RandomVerticalFlip(prob)
        - transforms.RandomRotation(degrees)
    - Be concious of what transforms you apply to the test set, you don't necessarily want to apply the same transforms to each. EX: Don't to flip images in the test set.
    - Generally don't shuffle for test set and test dataloader

2. Create the model. 
    - The "__init__" function:  
        - Add input channels (images == 3), number of classes for classification, etc.
        - If the model is sequential, use nn.Sequential() to set up the layers.
        - Determine if the model will do Classfication or regression. 
        - If you have a classification head, don't forget to flatten before you do the final fc layer.
    - The "forward()" function:
        - Call the forward pass, if using nn.sequential just call self.layers(X).
        - Should simply just take X, the input, and return the logits.
        
3. Set up the Loss function, Optimizer, Model:
    - Inititalize the model
    - Inititalize the Loss function
    - Initialize the Optimizer, passing in the model parameters and the Learning rate

4. Create Training Loop:
    - 





########################################################################################################
Architecture Design
Define the architecture based on the number of layers, types of functions, and connections.

Data Preparation
Prepare input and output data along with data loaders for efficiency.
Data normalization can be beneficial for many models.
Model Construction
Define a class to represent the neural network using torch.nn.Module. Use pre-built layers from torch.nn.

Loss and Optimizer Selection
Choose a loss function, such as Cross-Entropy for classification and Mean Squared Error for regression. Select an optimizer, like Stochastic Gradient Descent.

Training Loop
Iterate over batches of data.
Forward pass: Compute the model's predictions based on the input.
Backward pass: Calculate gradients and update weights to minimize the loss.
Model Evaluation
After training, assess the model's performance on a separate test dataset, typically using accuracy, precision, recall, or similar metrics.

Inference
Use the trained model to make predictions on new, unseen data

