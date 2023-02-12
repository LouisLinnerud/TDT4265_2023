import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel, cross_entropy_loss
from task2 import SoftmaxTrainer, calculate_accuracy


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    plt.figure(figsize=(20, 12))

    for i in range(4):
        if i != 3:
            continue
        if i == 0:
            val_legend = "Task 2 Model Validation loss"
            accuracy_val_legend = "Task 2 model validation accuracy"
            accuracy_tra_legend = "Task 2 model training accuracy"
        elif i == 1:
            use_improved_weight_init = True
            val_legend = "3a) Validation loss"
            accuracy_val_legend = "3a) model validation accuracy"
            accuracy_tra_legend = "3a) model training accuracy"
        elif i == 2:
            use_improved_sigmoid = True
            val_legend = "3b) Validation loss"
            accuracy_val_legend = "3b) model validation accuracy"
            accuracy_tra_legend = "3b) model training accuracy"
        elif i == 3:
            learning_rate = 0.02
            use_momentum = True
            val_legend = "3c) Validation loss"
            accuracy_val_legend = "3c) model validation accuracy"
            accuracy_tra_legend = "3c) model training accuracy"

        model = SoftmaxModel(
            neurons_per_layer,
            use_improved_sigmoid,
            use_improved_weight_init,
            use_relu)
        trainer = SoftmaxTrainer(
            momentum_gamma, use_momentum,
            model, learning_rate, batch_size, shuffle_data,
            X_train, Y_train, X_val, Y_val,
        )
        train_history, val_history = trainer.train(num_epochs)

        print("Final Train Cross Entropy Loss:",
              cross_entropy_loss(Y_train, model.forward(X_train)))
        print("Final Validation Cross Entropy Loss:",
              cross_entropy_loss(Y_val, model.forward(X_val)))
        print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
        print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

        plt.subplot(1, 2, 1)
        utils.plot_loss(val_history["loss"],
                        val_legend)
        plt.subplot(1, 2, 2)
        utils.plot_loss(val_history["accuracy"], accuracy_val_legend)
        utils.plot_loss(train_history["accuracy"], accuracy_tra_legend)

    plt.subplot(1, 2, 1)
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.ylim([0, .95])
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.ylim([0.6, 1.0])
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('task4a.png')
    plt.show()


if __name__ == "__main__":
    main()
