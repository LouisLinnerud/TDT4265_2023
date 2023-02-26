import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer


class Task3Model(nn.Module):

    def __init__(self, image_channels, num_classes):
        
        super().__init__()
        num_filters = 32
        self.num_classes = num_classes
        
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            
            # __ Layer 1 __
            #Conv2d 32 filters 
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            #Activation ReLU
            nn.ReLU(),
            #MaxPool2D 2x2
            nn.MaxPool2d(
                [2,2],
                stride=2
            ), 
            
            # __ Layer 2 __
            #Conv2d 64 filters 
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            #Activation ReLU
            nn.ReLU(),
            #MaxPool2D 2x2
            nn.MaxPool2d(
                [2,2],
                stride=2
            ),
            
            # __ Layer 3 __
            #Conv2d 128 filters
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            #Activation ReLU
            nn.ReLU(),
            #MaxPool2D 2x2
            nn.MaxPool2d(
                [2,2],
                stride=2
            )
        )

   
        self.num_output_features = 128*4*4
        self.num_hidden_units = 64
   
        # __ Layer 4 __
        self.classifier = nn.Sequential(
            #Fully connected 64 
            nn.Linear(self.num_output_features, self.num_hidden_units),
            #Activation ReLU
            nn.ReLU(),
            #Fully connected 10
            nn.Linear(self.num_hidden_units, num_classes)
        )

    def forward(self, x):
        
        batch_size = x.shape[0]
        x = self.feature_extractor(x)
        x = x.view(-1,self.num_output_features) # view(batch_size,-1) instead??
        x = self.classifier(x)
        
        out = x

        expected_shape = (batch_size, self.num_classes)
        
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = Task3Model(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    create_plots(trainer, "task2")
    print("task 3 ran")

if __name__ == "__main__":
    main()