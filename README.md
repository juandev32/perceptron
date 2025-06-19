# Implementing A Perceptron

I wrote a perceptron class to determine whether a sample's properties (60 features) were classified as a rock or metal. The peak testing accuracy was **88%** on the testing sample (20% of randomly chosen dataset **not included in training set**).

## What is a Perceptron?

### 5 Key Components

1. **Input Data**
   - Rows containing n features (60 in this case)
   - Feature values normalized between 0 and 1

2. **Weights**
   - Initialized to 0 for epoch 1
   - Computed as: sum of (weight[n] * feature[n]) for each sample
   - Processed as a vector operation

3. **Activation Function**
   - Sigmoid activation function: `1/(1 + e^(-z))` where z is the weighted sum
   - Decision boundary:
     - Returns 0 if output < 0.5
     - Returns 1 if output ≥ 0.5

4. **Weight Update**
   - Weights adjusted based on prediction accuracy
   - Update rule:  
     `self.synaptic_weights[i] += learning_rate * error * inputFeatures[i]`
   - Learning rate used: 0.01
   - One epoch = complete pass through all training samples (166 samples = 80% of data)

5. **Prediction on New Data**
   - After training (10 epochs in this implementation)
   - New samples processed through same weighted sum → sigmoid → classification
   - Testing accuracy evaluated on holdout set (20% of data)

## Implementation Details

### Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)
4. [Contact Information](#contact-information)

### Installation
No external dependencies required.  
*Core libraries used:*
- `random` for sampling
- `csv` for data loading

### Usage
File: `Perceptron_Tester.py`

**Class Functions:**
- Input validation
- Sigmoid activation
- Random 80/20 train-test split
- Accuracy evaluation

**Output:**
- Terminal displays both training and testing accuracy
- Helps detect overfitting (if training accuracy >> testing accuracy)

### Features
- Compatible with any CSV dataset (modify feature fields as needed)
- Designed for linearly separable problems
- Note: Complex problems require more advanced neural networks (see other projects)

### Contact Information
**Email:** [juandev32@gmail.com](mailto:juandev32@gmail.com)  
**LinkedIn:** [Juan Chavira's Profile](https://www.linkedin.com/in/juan-chavira/)