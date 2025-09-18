# Constructing a Pipeline from Scratch

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
    - The __init__ function:  
        - Add input channels (images == 3), number of classes for classification, etc.
        - If the model is sequential, use nn.Sequential() to set up the layers.
        
3. 
