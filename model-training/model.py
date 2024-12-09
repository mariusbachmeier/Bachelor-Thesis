import torch.nn as nn


class MultiNet(nn.Module):
    def __init__(self, backbone, num_features: int, num_class_dict: dict, device: str):
        """
        init function of the model specifying its structure containing a shared backbone and multiple separate heads
        according to the multi-domain multi-task training paradigm.

        :param backbone: holds the backbone of the model, a model that got its classification layer removed
        :param num_features: the number of features the last layer, so the classification layer, will receive as input
        :param num_class_dict: holds the number of output features for the classification layers that are about to be
        instantiated
        :param device: specifies on which device the classification layers will be stored
        """
        super(MultiNet, self).__init__()
        self.backbone = backbone
        for key, value in num_class_dict.items():
            print(f"Creating a Head for dataset {key} called fc{key} with input num features: {num_features} and "
                  f"output num features: {value}")
            setattr(self, f'fc{key}', nn.Linear(in_features=num_features, out_features=value).to(device))

    def forward(self, x, dataset_name):
        """
            Performs the forward pass of the model.

            :param x: image data
            :param dataset_name: Specifies the dataset for which the forward pass should be performed.
            :return: The model's predictions as probabilities.
        """
        x = self.backbone(x)
        fc_layer = getattr(self, f'fc{dataset_name}')
        x = fc_layer(x)
        return x

    def train_step(self, images, labels, dataset_name: str, optimizer, criterion, weighted_loss: bool,
                   weights_dict: dict):
        """
        Performs the training step of the model,

        param images: Holds the image tensors.
        param labels: Holds the ground truth labels.
        param dataset_name: Holds the name of the dataset as a string, used for indexing.
        param optimizer: The optimizer of the model.
        param criterion: loss calculation function
        param weighted_loss: Whether the loss is to be weighted.
        param weights_dict: The weights in case the loss is to be weighted.
        :return: The loss and the model's predictions as probabilities.
        """
        # Set all parameters' requires_grad to False first
        for param in self.parameters():
            param.requires_grad = False

        # Activate only the selected layer and backbone
        self.backbone.requires_grad_(True)
        getattr(self, f'fc{dataset_name}').requires_grad_(True)

        # Forward pass
        outputs = self.forward(images, dataset_name)
        if weighted_loss:
            loss = weights_dict[dataset_name] * criterion(outputs, labels)
        else:
            loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        outputs_detached = outputs.clone().detach()

        # Update only the parameters that were selected for this batch
        optimizer.step()

        optimizer.zero_grad()

        return loss.item(), outputs_detached
