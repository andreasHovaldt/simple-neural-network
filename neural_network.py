import numpy as np, random, os
from keras.datasets import mnist


# Knowledge gained from:
# http://neuralnetworksanddeeplearning.com/chap1.html
# http://neuralnetworksanddeeplearning.com/chap2.html


#----------------------------- GLOSSARY -----------------------------#
#
# C: Cost, the mean squared error (MSE) of the network output to the desired/correct output
#
# z: "Weighted input", z = w^l * a^(l-1) + b^l
# 
# a: Activation(s), the neuron outputs, a = sigmoid(z), 
#    a can also be seen as the input to the next layer of neurons
#
# y: Desired output, the correct output to a given input, used in cost function and to train network
#
# learning_rate: How fast and erratic the network should correct itself, lower values take longer, 
#                while higher values might make the correction unstable. Also known as eta.
#
# weights: Size is a matrix where, columns are are previous layer's number of neurons, and rows are current/next layer's number of neurons   
#
# biases: Size is a vector with amount of indices equal to current layer's number of neurons 
#
# training data: Size should be a (x,y) tuple, where x is a (n,1) vector input, and y is a (m,1) desired output vector corresponding to the input  
#
# neuron error: delta, delta_L, Determines how much each individual neuron in the network influences the error in the final output 
#               Pretty sure its the gradient -> del C / del delta_L, gradient then looks at how much each neuron influences the error of the final output
#
# delta biases: The rate of change of the biases wrt. cost
#
# delta weights: The rate of change of the weights wrt. cost
#
# Gradient: Rate of change of x wrt. y
# 
#
#
#


#----------------------------- FUNCTIONS -----------------------------#
def sigmoid(z):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-z)) # Calculating the sigmoid function (scale value into range from 0 to 1)

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z)) # Calculating the derivative of the sigmoid function

def gradient(z,a, a_1, y):
    """The gradient describes the direction and magnitude of the function ascent"""
    weight_sensitivity = a_1 * sigmoid_prime(z) * (2 * (a - y)) # Calculating the partial derivative of the cost function wrt. weight
    bias_sensitivity = 1 * sigmoid_prime(z) * (2 * (a - y)) # ----||---- wrt. bias
    return [weight_sensitivity, bias_sensitivity]

def load_mnist(train_amount = 60000, test_amount = 10000): #TODO: Normalize pixel values between 0-1, will improve network prediction
    (train_X, train_y), (test_X, test_y) = mnist.load_data() # Load in data
    
    # Define how much of the dataset is to be used
    train_X = train_X[:train_amount]
    train_y = train_y[:train_amount]
    
    test_X = test_X[:test_amount]
    test_y = test_y[:test_amount]
    
    
    # Define arrays for storing the reshaped data
    train_X_reshaped = []
    train_y_reshaped = []
    test_X_reshaped = []
    test_y_reshaped = []
    
    # Reshape data
    for X, y, i in zip(train_X, train_y, range(len(train_y))):
        train_X_reshaped.append(np.reshape(X,(28*28,1))) # Reshape/flatten data input to match the desired input for the network
        #np.savetxt(f"mnist_testing/mnist_train_X_{i}.txt",train_X_reshaped[i])
        
        temp_y = np.zeros((10,1)) # Inputs have 10 possible outcomes, 0->9
        temp_y[y] = 1
        train_y_reshaped.append(temp_y)
        #np.savetxt(f"mnist_testing/mnist_train_y_{i}.txt",train_y_reshaped[i])
        
    for X, y, i in zip(test_X, test_y, range(len(test_y))):
        test_X_reshaped.append(np.reshape(X,(28*28,1))) # Reshape/flatten data input to match the desired input for the network
        #np.savetxt(f"mnist_testing/mnist_train_X_{i}.txt",train_X_reshaped[i])
        
        temp_y = np.zeros((10,1)) # Inputs have 10 possible outcomes, 0->9
        temp_y[y] = 1
        test_y_reshaped.append(temp_y)
        #np.savetxt(f"mnist_testing/mnist_train_y_{i}.txt",train_y_reshaped[i])

    return train_X_reshaped, train_y_reshaped, test_X_reshaped, test_y_reshaped

#----------------------------- CLASS -----------------------------#
class Network():
    def __init__(self, size, debug=False) -> None:
        """
        Size is the structure of the network [input size, hidden layer 1 size, ... , output size]
            Example: [10, 4, 4, 3] -> Input has 10 neurons, network has 2 hidden layers with 4 neurons each, network has 3 outputs
        
        """
        self.input_size = size[0] # Size of the input vector of the network
        self.size = size
        self.layers = len(size) # Number of layers in the network
        self.output_size = size[self.layers - 1] # Size of the output vector of the network
        
        # Generates random weight values with matrix sizes equal to the number of inputs to each neuron, 
        # by the number of neurons in the given layer
        self.weights = [np.random.randn(y, x) for x, y in zip(size[:-1], size[1:])]
        
        # Generates random bias values with vector sizes equal to the number of outputs each neuron in the given layer has
        self.biases = [np.random.randn(y, 1) for y in size[1:]]
        
        
        self.debug = debug
        #self.total_average_cost = None
        
    

    
    def feedforward(self, a):
        """Go through the network while computing and saving each layer's z- and neuron actionvation values
        Args:
            a: Input data, should be a (n,1) vector, where n equals the number of neurons in the first layer of the network (input layer)
        """
            
        zs = [] # Used to store all z's
        activations = [a] # Used to store all neuron activations
        
        
        for L in range(self.layers - 1): # Go through each layer (-1 because we skip the first layer, which is the input layer)
            # For each iteration we want to use the previous neuron's outputs
            z = np.dot(self.weights[L],a) + self.biases[L] # Calculate z
            a = sigmoid(z) # Apply the sigmoid function to z
            
            zs.append(z) # Save this layer's z's
            activations.append(a) # Save this layer's neuron activations
                        
            # debug
            if self.debug: 
                np.savetxt(f"input_after_layer_{L}.txt", a)
                np.savetxt(f"zs_after_layer_{L}.txt", z)
            
        return zs, activations

    
    
    def backpropagation(self, input, desired_output, display_cost = False):
        """Using back propagation the function is able to calculate the rate of change for Cost, wrt. the weights and biases

        Args:
            input: Input data to send through the network, should be in the form of a (n,1) vector, where n is the number of input neurons
            desired_output: The desired output of the network, should be in the form of a (m,1 vector), where m is the number of output neurons
            display_cost (bool): If true, displays cost for every iteration of the training data

        Returns:
            tuple: Rate of change for Cost, wrt. the weights and biases
        """
        
        # Feedforward -> Send input through the network to obtain z's and neuron activations for all layers
        zs, activations = self.feedforward(input)
        
        # Brackprop
        delta_biases = [np.zeros(biases.shape) for biases in self.biases] # Create empty array for storing the error for each bias (Gradient/nabla for biases)
        delta_weights = [np.zeros(weights.shape) for weights in self.weights] # Create empty array for storing the error for each weight (Gradient/nabla for weights)
        
        delta_L = (activations[-1] - desired_output) * sigmoid_prime(zs[-1]) # EQ 30, chap 2 -> neuron error in output layer (size: (n,1), where n is no. of neurons in layer)
        delta_biases[-1] = delta_L # EQ 31, chap 2
        delta_weights[-1] = np.dot(delta_L, np.transpose(activations[-2])) # EQ 32, chap 2 -> It is a little different from the book, but using inspiration from the books code, this way of calculating it achieves the correct matrix with the same sizes as the weight matrices
        
        
        for L in range(2, self.layers): # Starts at two, since we have already calculated error for the output layer
            
            delta_L = np.dot(np.transpose(self.weights[-L + 1]), delta_L) * sigmoid_prime(zs[-L]) # EQ BP2, chap 2
            # Calculates the neuron errors for current layer L, starts off by using delta_L calculated from the output layer, 
            #   then updates this delta_L for each interation, thus calculating the neuron error further and further into the network 
            # Weights has + 1, because then we start with [-1], and since the last weight matrix relates back to the second last layer, 
            #   we want the last weight matrix and not the second last, since that one would relate back to the third last layer
            
            delta_biases[-L] = delta_L # EQ 31, chap 2
            delta_weights[-L] = np.dot(delta_L, np.transpose(activations[-L - 1])) # EQ 32, chap 2
        
        if display_cost: cost = self.cost_function(activations[-1],desired_output) # Calculate the cost of the network output, compared to the desired output
        else: cost = 0
            
        return (delta_weights, delta_biases, cost)
        
        
        
        
    def train_network(self, X, y, learning_rate, display_cost = False):
        """Update weights and biases based on given training_data and learning_rate

        Args:
            training_data (tuple): Should be data (batch) consisting of training inputs and their corresponding desired outputs (x, y)
            learning_rate (int): How fast and erratic the network should correct itself, lower values take longer, while higher values might make the correction unstable
            display_cost (bool): If true, displays cost for every iteration of the training data
        """
        
        ## Back propagation
        # Create empty arrays, which will be used for storing the gradients calculated for each iteration of the backpropagation
        weight_gradients = [np.zeros(weights.shape) for weights in self.weights]
        bias_gradients = [np.zeros(biases.shape) for biases in self.biases]
        
        
        if display_cost: costs = [] # Create empty array for storing costs
        
        # Backpropagate for each input and the corresponding desired output
        for input, desired_output in zip(X,y):
            delta_weights, delta_biases, cost = self.backpropagation(input, desired_output, display_cost) # Backpropagate for input
            
            weight_gradients = [prev_w + new_w for prev_w, new_w in zip(weight_gradients, delta_weights)] # Save backpropagation for input into previously determined weight gradients
            bias_gradients = [prev_b + new_b for prev_b, new_b in zip(bias_gradients, delta_biases)] # -----||----- bias gradients
            
            if display_cost: costs.append(cost) # Save the cost for each input
            
        if display_cost: print(np.sum(costs) / len(costs)) # Calculate the mean cost for the batch
            
        
        ## Weight and bias updating
        #self.weights = self.weights - ((learning_rate / len(X)) * weight_gradients)
        #self.biases = self.biases - ((learning_rate / len(X)) * bias_gradients)
        
        # Every weight and bias is updated according to their respective gradients, eq 20+21 chap. 1
        self.weights = [w - (learning_rate/len(X)) * w_g for w, w_g in zip(self.weights, weight_gradients)]
        self.biases = [b - (learning_rate/len(X)) * b_g for b, b_g in zip(self.biases, bias_gradients)]
        
    
    
    def stochastic_gradient_decent(self, training_data, epochs, training_batch_size, learning_rate, display_cost = False):
        """Traing the neural network using stichastic gradient decent

        Args:
            training_data (tuple): Should be data consisting of training inputs and their corresponding desired outputs (x, y)
            epochs (int): The number of training cycles done on the training data
            mini_batch_size (int): The number of samples to train on before correcting weights and biases
            learning_rate (int): How fast and erratic the network should correct itself, lower values take longer, while higher values might make the correction unstable
        """

        for epoch in range(epochs):
            #print(f"Training progress: Epoch {epoch+1} of {epochs}", end='\r')
            print(f"Initiating training on epoch {epoch+1} of {epochs}")
            
            #random.shuffle(training_data) # Randomly shuffle around the training data TODO: Implement shuffle, while making sure input and desired output still match
            
            n = len(training_data[0]) # Get number of training inputs (no. of images to train on)
            input_batches = [training_data[0][i:i+training_batch_size] for i in range(0, n, training_batch_size)] # Split training data into batches of defined training_batch_size
            output_batches = [training_data[1][i:i+training_batch_size] for i in range(0, n, training_batch_size)] # Split training data into batches of defined training_batch_size
            
            
            n_batches = len(input_batches) # Number of batches in one epoch, used to calculate progress within an epoch
            for input_batch, output_batch, i in zip(input_batches, output_batches, range(n_batches)): # Go through each training batch
                self.train_network(input_batch, output_batch, learning_rate, display_cost) # Update the weights and biases using the current training_batch
                progress_percent = round(i / n_batches *100,1)
                if (progress_percent % 10) == 0: print(f"Training progress: Epoch {epoch+1} of {epochs} - {progress_percent}%")
             
        print("Training done!")
    

    def cost_function(self, output_vector, desired_output_vector): # aka Loss or objective function
        return np.sum((output_vector - desired_output_vector)**2) # Calculate total cost of network output
    
    
    def eval_performance(self, test_X, test_y):
        
        correct = 0
        
        for X, y in zip(test_X, test_y):
            _, a = self.feedforward(X)
            if np.argmax(a[-1]) == np.argmax(y): correct += 1
            
        return correct / len(test_X) # Performance
    
    def save_network(self,performance):
        """Calling this function saves the characteristics, performance, weights and biases of the network into the current directory"""
        folder_path = os.path.join(os.getcwd(),f"Network_{self.size}") # Get current directory and add name of new folder to the current directory path
        
        # Saving the network weights and biases as .txt files within the network folder
        weights_path = os.path.join(folder_path, "Weights") # Directory for weights
        biases_path = os.path.join(folder_path, "Biases") # Directory for biases
        
        os.makedirs(weights_path, exist_ok=True) # Create folder for saving network weights
        os.makedirs(biases_path, exist_ok=True) # Create folder for saving network biases
        
        for L in range(len(self.weights)):
            np.savetxt(f"{weights_path}/layer_{L}",self.weights[L])
        
        for L in range(len(self.biases)):
            np.savetxt(f"{biases_path}/layer_{L}",self.biases[L])
        
        # Saving the network characteristics as a .txt file within the network folder
        try: f = open(os.path.join(folder_path,"Network_stats.txt"),"x") # If file doesn't exist it will be create
        except: f = open(os.path.join(folder_path,"Network_stats.txt"),"w") # If file does exist it will be emptied and rewritten to

        f.write(f"Network structure: {self.size}\nHidden layers: {self.layers - 2}\nHidden neurons: {np.sum(self.size[1:-1])}\nPerformance: {performance}%")
        f.close()
    








def main():
    (train_X, train_y, test_X, test_y) = load_mnist(train_amount=60000, test_amount=10000)

    ntwk = Network([784,50,50,50,50,50,10])
    ntwk.stochastic_gradient_decent(training_data=(train_X,train_y),epochs=500,training_batch_size=100,learning_rate=0.5,display_cost=False)
    ntwk_performance = ntwk.eval_performance(test_X=test_X, test_y=test_y)
    print(f"Network performance: {ntwk_performance * 100}%")
    
    ntwk.save_network(ntwk_performance*100)
    

if __name__ == "__main__":
    main()








