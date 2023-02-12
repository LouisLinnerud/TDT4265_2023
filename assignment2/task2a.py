import numpy as np
import utils
import typing
np.random.seed(1)


def pre_process_images(X: np.ndarray, mu: float = 33.55274553571429, sigma: float = 78.87550070784701):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    X_norm = (X - mu)/sigma
    X_biased = np.array([np.append(x, 1) for x in X_norm])
    return X_biased


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    Cn = - np.sum(targets*np.log(outputs), axis=1, keepdims=True)
    loss = 1/len(targets)*np.sum(Cn)

    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    return loss


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3b hyperparameter
                 use_improved_weight_init: bool,  # Task 3a hyperparameter
                 use_relu: bool  # Task 4 hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.number_of_layers = len(neurons_per_layer)
        print('Number of layers =', self.number_of_layers)
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_relu = use_relu
        self.use_improved_weight_init = use_improved_weight_init

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            if self.use_improved_weight_init:
                w = np.random.normal(0, 1/np.sqrt(prev), w_shape)
            else:
                w = np.random.uniform(-1, 1, w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]
        # print('yes')
        # [print(np.shape(w)) for w in self.ws]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...

        # Execute forward pass for all hidden layers
        activation = X
        self.activations = [X]
        self.zs = []
        # - 1 since last layers should use softmax, not sigmoid
        for layerIndex in range(self.number_of_layers - 1):
            w = self.ws[layerIndex]
            z = activation.dot(w)
            self.zs.append(z)
            if self.use_improved_sigmoid:
                activation = self.improved_sigmoid(z)
            else:
                activation = self.sigmoid(z)
            self.activations.append(activation)
        # print('activation shape = ', np.shape(activation))

        # Fetch weights to final layer
        w = self.ws[-1]
        # Calculate zk (z of final layer):
        zk = activation.dot(w)
        # Apply softmax to zk to get final activation
        activation = self.softmax(zk)
        self.activations.append(activation)
        return activation

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def improved_sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.7159*np.tanh(2./3.*z)

    def improved_sigmoid_prime(self, z: np.ndarray) -> np.ndarray:
        # See https://www.cuemath.com/calculus/derivative-of-hyperbolic-functions/ for derivative
        return 1.7159 * 2./3. / (np.cosh(2./3.*z))**2

    def sigmoid_prime(self, z: np.ndarray) -> np.ndarray:
        # see https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
        sigma = self.sigmoid(z)
        return sigma*(1-sigma)

    def softmax(self, z: np.ndarray) -> np.ndarray:
        return np.exp(z)/np.sum(np.exp(z), axis=1, keepdims=True)

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        self.grads = []

        # Calculate output error
        delta_k = -(targets - outputs)
        L = self.number_of_layers - 1
        # print('L =', L)
        deltas = [[] for i in range(L + 1)]
        deltas[-1] = delta_k
        # print(len(deltas))
        # print(deltas)
        # Backpropegate starting from l = L - 1, L - 2, ... 1 (where 0 counts as layer due to indexing in python)
        for layerIndex in range(L - 1, -1, - 1):
            # print('L=', L)
            # print('index =', layerIndex)
            # l = L - layerIndex
            # print('l=', l)
            zl = self.zs[layerIndex]
            # get sigmoid primed of z at the current layer
            if self.improved_sigmoid:
                derivative_zl = self.improved_sigmoid_prime(zl)
            else:
                derivative_zl = self.sigmoid_prime(zl)
            # print('der shape:', np.shape(derivative_zl))
            delta_l_pluss_1 = deltas[layerIndex + 1]
            weight_l_pluss_1 = self.ws[layerIndex + 1]
            # print(np.shape(delta_l_pluss_1.dot(weight_l_pluss_1.T)))
            delta = (delta_l_pluss_1.dot(weight_l_pluss_1.T)) * derivative_zl
            deltas[layerIndex] = delta

        # Set the gradients
        # Note that the indexing of self.activations includes the input layer, dividing by X.shape[0] to get the average
        for l in range(L + 1):
            grad = self.activations[l].T.dot(deltas[l])/X.shape[0]
            # print('gradshape =', np.shape(grad))
            self.grads.append(grad)
        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    # def softmax_prime(self, z: np.ndarray) -> np.ndarray:
    #     pass

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # TODO: Implement this function (copy from last assignment)
    Y_encoded = np.array([[0]*num_classes for i in range(len(Y))])
    for i in range(len(Y)):
        y = Y[i][0]
        Y_encoded[i][y] = 1
    return Y_encoded


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited.
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        # print('layer_idx starting:', layer_idx)
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**1,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


def main():
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    # print(np.shape(X_train))
    mu = np.mean(X_train)
    std = np.std(X_train)
    print(mu)
    print(std)
    X_train = pre_process_images(X_train, mu, std)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785, \
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_relu = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)
    # for grad in model.grads:
    #     print('shape of grad: ', np.shape(grad))

    gradient_approximation_test(model, X_train, Y_train)


if __name__ == "__main__":
    main()
