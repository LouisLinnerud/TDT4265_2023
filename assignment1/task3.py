import time
import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    logits = model.forward(X)
    # Convert [0,0,0,1,0,0,0,0,0] to 4 etc.
    target_int = np.argmax(targets, axis=1)
    output_int = np.argmax(logits, axis=1)
    accuracy = np.mean(target_int == output_int)
    # This function was optimized using chatGPT
    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        outputs = self.model.forward(X_batch)
        self.model.backward(X_batch, outputs, Y_batch)
        self.model.w = self.model.w - self.learning_rate * self.model.grad
        loss = cross_entropy_loss(Y_batch, outputs)
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(self.Y_val, logits)

        accuracy_train = calculate_accuracy(
            self.X_train, self.Y_train, self.model)
        accuracy_val = calculate_accuracy(
            self.X_val, self.Y_val, self.model)
        return loss, accuracy_train, accuracy_val


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    # num_epochs = 50 as default, changing it to test for overfitting
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda=0)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    weight = val_history['w_min']

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([0.2, .8])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    # Train a model with L2 regularization (task 4b)

    model1 = SoftmaxModel(l2_reg_lambda=.1)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)

    weight1 = val_history_reg01['w_min']

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model1.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model1.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model1))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model1))

    plt.ylim([0.7, .95])
    utils.plot_loss(train_history_reg01["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history_reg01["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.5, .95])
    utils.plot_loss(train_history_reg01["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history_reg01["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    # You can finish the rest of task 4 below this point.

    # Plotting of softmax weights (Task 4b)

    # TODO: Reshape weights from (785,10) to 10 images of 28x28
    image_data = []
    image_data1 = []

    # Remove last weight and reshape to 28x28 for the weight per digit
    for i in range(len(weight[0])):
        image_data.append(np.reshape(weight[:, i][:-1], (28, 28)))
        image_data1.append(np.reshape(weight1[:, i][:-1], (28, 28)))

    # f, axarr = plt.subplots(2, 10)
    # for ax in axarr:
    #     for a in ax:
    #         a.set_xticklabels([])
    #         a.set_yticklabels([])
    # for i in range(len(weight[0])):
    #     axarr[0, i].imshow(image_data[i], cmap='gray')
    #     axarr[1, i].imshow(image_data1[i], cmap='gray')
    # plt.show()
    # print(np.shape(image_data[0]))
    # print([image for image in image_data])

    weight_visualize = np.concatenate([image for image in image_data], axis=1)
    weight_visualize1 = np.concatenate(
        [image for image in image_data1], axis=1)

    weight_visualize_total = np.concatenate(
        (weight_visualize, weight_visualize1), axis=0)

    plt.imsave("task4b_softmax_weight.png", weight_visualize, cmap="gray")
    plt.imsave("task4b_softmax_weight1.png", weight_visualize1, cmap="gray")
    plt.imsave("task4b_softmax_total.png", weight_visualize_total, cmap="gray")
    weight_visualize_total2 = np.concatenate(
        (weight_visualize, weight_visualize1*3), axis=0)
    plt.imsave("task4b_softmax_total2.png",
               weight_visualize_total2, cmap="gray")

    # plt.imsave("task4b_softmax_weight1.png", weight1.T, cmap="gray")

    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [1, .1, .01, .001]
    models = []
    trainers = []
    train_histories = []
    val_histories = []
    weights = []
    l2norms = []
    for i in range(len(l2_lambdas)):
        l2_reg_lambda = l2_lambdas[i]
        model = SoftmaxModel(l2_reg_lambda=l2_reg_lambda)
        models.append(model)
        trainer = SoftmaxTrainer(model, learning_rate, batch_size, shuffle_dataset,
                                 X_train, Y_train, X_val, Y_val)
        trainers.append(trainer
                        )
        train_history, val_history = trainer.train(num_epochs)
        train_histories.append(train_history)
        val_histories.append(val_history)
        weight = val_history['w_min']
        weights.append(weight)
        l2norm = np.sum(weight**2)
        l2norms.append(l2norm)
        print()
        print(f"Results for model with L2_lambda = {l2_reg_lambda}:")
        print("Final Train Cross Entropy Loss:",
              cross_entropy_loss(Y_train, model.forward(X_train)))
        print("Final Validation Cross Entropy Loss:",
              cross_entropy_loss(Y_val, model.forward(X_val)))
        print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
        print("Final Validation accuracy:",
              calculate_accuracy(X_val, Y_val, model))

        # utils.plot_loss(train_history_reg01["accuracy"], f"Training Accuracy")
        utils.plot_loss(
            val_history["accuracy"], f"Validation Accuracy for lam = {l2_reg_lambda}")
    plt.ylim([.0, .95])

    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show()

    # Task 4d - Plotting of the l2 norm for each weight

    plt.plot(l2_lambdas, l2norms, label='lambda_norm')
    plt.xlabel('Lambda values')
    plt.ylabel('Norm of weight')
    plt.legend()
    plt.savefig("task4d_l2_reg_norms.png")
    plt.show()


if __name__ == "__main__":
    main()
