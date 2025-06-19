from csv import reader					# reader object reads a csv file line by line
from random import seed					# seeds the random number generator
from random import randrange			# returns a random value in a specified range
from random import sample


class Perceptron(object):
	# Create a new Perceptron
	# 
	# Params:	bias -	arbitrarily chosen value that affects the overall output
	#					regardless of the inputs
	#
	#ynaptic_weights -	list of initial synaptic weights for this Perceptron
	def __init__(self, bias, synaptic_weights):
		
		self.bias = bias							#w0 (bias) should be initialized to start with 0 adjust 
		self.synaptic_weights = synaptic_weights

	# Activation function
	#	Quantizes the induced local field
	#
	# Params:	z - the value of the indiced local field
	#
	# Returns:	an integer that corresponds to one of the two possible output values (usually 0 or 1)
	# 
	# Because the case exists where, when computing the error in the train function
	# 1. Actual label T or F is 0 (can be either since enumerated and training data is selected randomly)
	# 2. The Predicted value can be 1
	# 3. actual label (0) - predicted label (1) would cause negative products to floor to 0 if using ReLu activation function
	# 4. Sigmoid or tanH functions would be effective activation functions, sigmoid just required one input (z), tanh(z) would require 2 
	# 5. if using tanH then switch threshold for neuron to return 0 for products <0 and 1 for >=0
	def activation_function(self, z):
		"""
		Sigmoid activation function
		1/(1+2.71828**(-z)
		"""
		return 1 if 1/(1+2.71828**(-z))>=.5 else 0		

	# Compute and return the weighted sum of all inputs (not including bias)
	#
	# Params:	inputs - a single input vector (which may contain multiple individual inputs)
	#
	# Returns:	a float value equal to the sum of each input multiplied by its
	#			corresponding synaptic weight
	def weighted_sum_inputs(self, inputs):
		if(len(inputs)!=len(self.synaptic_weights)):
			raise ValueError("Mismatch in number of inputs and synapticWeights")

		weightedSum=0.0000	#initialized weighted sum to float 0

		for i in range(len(inputs)):
			weightedSum+=self.synaptic_weights[i]*inputs[i]
				#multiply input by synaptic weights 
				#normally the inputs are transposed so the dot product is correct

		return weightedSum

	# Compute the induced local field (the weighted sum of the inputs + the bias)
	#
	# Params:	inputs - a single input vector (which may contain multiple individual inputs)
	#
	# Returns:	the sum of the weighted inputs adjusted by the bias
	def induced_local_field(self, inputs):
		return self.weighted_sum_inputs(inputs)+self.bias

	# Predict the output for the specified input vector
	#
	# Params:	input_vector - a vector or row containing a collection of individual inputs
	#
	# Returns:	an integer value representing the final output, which must be one of the two
	#			possible output values (usually 0 or 1)
	def predict(self, input_vector):
		#"predictions" are the induced local field passed through the activation function to determine if the neuron will return 1 or 0
		inducedLocalField= self.induced_local_field(input_vector)
		return self.activation_function(inducedLocalField)

	# Train this Perceptron
	#
	# Params:	training_set - a collection of input vectors that represents a subset of the entire dataset
	#			learning_rate_parameter - 	the amount by which to adjust the synaptic weights following an
	#										incorrect prediction
	#			number_of_epochs -	the number of times the entire training set is processed by the perceptron
	#
	# Returns:	no return value

	def train(self, training_set, learning_rate_parameter, number_of_epochs):
		for _ in range(number_of_epochs + 1):

			for row in range(len(training_set)):
				# Extract input features and desired output
				inputFeatures = training_set[row][:60]
				actualOutput = self.predict(inputFeatures)

				#compute error (1 -1 0) for determining equation for change in synaptic weight next step
				error = training_set[row][-1] - actualOutput

				# Update weights for each input feature
				for i in range(len(self.synaptic_weights)):
					#THIS IS THE PERCEPTRON "LEARNING" FUNCTION
					#w(n+1)=w(n)+learningRate*[desired-actual]*input
					self.synaptic_weights[i] += learning_rate_parameter * error * inputFeatures[i]

	# Test this Perceptron
	# Params:	test_set - the set of input vectors to be used to test the perceptron after it has been trained
	#
	# Returns:	a collection or list containing the actual output (i.e., prediction) for each input vector
	def test(self, test_set,training_set):
		#I did both training and testing set to notice overfitting while i change epochs & learning rate parameter
		#compute the accuracy on the training set
		trainCorrect=0
		trainTotal=len(training_set)
		print("Computing training accuracy")
		for inputs in training_set:
			actual=inputs[-1]
			prediction=self.predict(inputs[:-1])
			print("Actual: {} prediction: {}".format(actual, prediction))
			if prediction==actual:
				trainCorrect+=1

		trainAccuracy= trainCorrect/trainTotal
		
		#computer accuracy on testing set
		testCorrect=0
		testTotal=len(test_set)
		print("Computing testing accuracy")
		for inputs in test_set:
			actual=inputs[-1]
			prediction=self.predict(inputs[:-1])
			print("Actual: {} prediction: {}".format(actual, prediction))
			if prediction==actual:
				testCorrect+=1

		testAccuracy= testCorrect/testTotal

		print("Training accuracy: {}".format(trainAccuracy))
		print("Testing accuracy: {}".format(testAccuracy))
		#return accuracy

class test:
    # Dataset Functions

	# Load the CSV file containing the inputs and desired outputs
	#
	#	dataset is a 2D matrix where each row contains 1 set of inputs plus the desired output
	#		-for each row, columns 0-59 contain the inputs as floating point values
	#		-column 60 contains the desired output as a character: 'R' for Rock or 'M' for Metal
	#		-all values will be string values; conversion to appropriate types will be necessary
	#		-no bias value is included in the data file
	def load_csv(filename):
		# dataset will be the matrix containing the inputs
		dataset = list()

		# Standard Python code to read each line of text from the file as a row
		with open(filename, 'r') as file:
			csv_reader = reader(file)
			for row in csv_reader:
				if not row:
					continue

				# add current row to dataset
				dataset.append(row)

		return dataset

	# Convert the input values in the specified column of the dataset from strings to floats
	def convert_inputs_to_float(dataset, column):
		for row in dataset:
			row[column] = float(row[column].strip())

	# Convert the desired output values, located in the specified column, to unique integers
	# For 2 classes of outputs, 1 desired output will be 0, the other will be 1
	def convert_desired_outputs_to_int(dataset, column):
		# Enumerate all the values in the specified column for each row
		class_values = [row[column] for row in dataset]

		# Create a set containing only the unique values
		unique = set(class_values)

		# Create a lookup table to map each unique value to an integer (either 0 or 1)
		lookup = dict()
		for i, value in enumerate(unique):
			lookup[value] = i

		# Replace the desired output string values with the corresponding integer values
		for row in dataset:
			row[column] = lookup[row[column]]
		
		return lookup

	# Load the dataset from the CSV file specified by filename
	def load_dataset(filename):
		# Read the data from the specified file
		dataset = test.load_csv(filename)

		# Convert all the input values form strings to floats
		for column in range(len(dataset[0])-1): 
			test.convert_inputs_to_float(dataset, column)

		# Convert the desired outputs from strings to ints
		test.convert_desired_outputs_to_int(dataset, len(dataset[0]) - 1)

	# Create the training set
	#	-Training set will consist of the specified percent fraction of the dataset
	#	-num inputs you decide to use for the training set, and how you choose
	#	 those values, optimize this
	#
	# Params:	dataset - the entire dataset
	#
	# Returns:	a matrix, or list of rows, containing only a subset of the input
	#			vectors from the entire dataset
	def create_training_set(dataset):
		print("create training set output check if last element")
		
		sampleSize=.8
		trainingSetSize=sampleSize*len(dataset)
		trainingSet=sample(dataset,int(trainingSetSize))

		testingSet = [sample for sample in dataset if sample not in trainingSet]
		return trainingSet,testingSet

#Create, train, test perceptron

#Acquire the dataset
dataset = test.load_csv("sonar_all-data.csv")

#Convert the string input values to floats
for column in range(0,60):
	test.convert_inputs_to_float(dataset,column)

#Convert the desired outputs to int values
test.convert_desired_outputs_to_int(dataset,60)

#Create the training set
trainingSet,testingSet=test.create_training_set(dataset)

#Create the perceptron

#print(len(trainingSet[0])-1)
perceptron=Perceptron(0,[0]*(len(trainingSet[0])-1))

#Train the perceptron
perceptron.train(trainingSet,.01,10)
#Test the trained perceptron
perceptron.test(testingSet,trainingSet)

#Display the test results and accuracy of the perceptron