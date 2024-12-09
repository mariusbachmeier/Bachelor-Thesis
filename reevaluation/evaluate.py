"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
The script reevaluates the models that were trained as part of "Rethinking Model Prototyping through the MedMNIST+
Dataset Collection" to obtain the performance of them for the evaluation metrics AUC, balanced accuracy and
cohen's kappa
"""

from utils import (
    calculate_passed_time, seed_worker, extract_embeddings,
    extract_embeddings_alexnet, extract_embeddings_densenet,
    knn_majority_vote, getACC, getAUC, get_Balanced_ACC,
    get_Cohen, get_Precision, get_ACC_kNN, get_Balanced_ACC_kNN,
    get_Cohen_kNN, get_Precision_kNN
)

# Import packages
import os
import argparse
import yaml
import torch
import torch.nn as nn
import timm
import time
import medmnist
import random
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models import alexnet, AlexNet_Weights
from medmnist import INFO
from sklearn.neighbors import NearestNeighbors


def evaluate(model_path: str, training_procedure: str, architecture: str, img_size: int, num_classes: int,
             device: str, task: str, save_path, train_loader: DataLoader, test_loader: DataLoader):
    """
    Evaluate a model on the specified dataset.

    :param model_path: Where the model which is about to be evaluated is located.
    :param training_procedure: endToEnd, linearProbing or kNN.
    :param architecture: The Architecture of the model.
    :param img_size: The size of the images that should be loaded.
    :param num_classes:
    :param device: Specifies on which device the cuda device.
    :param task: Describes the task corresponding to the dataset.
    :param save_path: where the should be saved.
    :param train_loader: Holds the training dataloader.
    :param train_loader: Holds the test dataloader.


    """

    # Start code
    start_time = time.time()

    # Load the trained model
    print("\tLoad the trained model ...")
    if architecture == 'alexnet':
        model = alexnet(weights=AlexNet_Weights.DEFAULT)

        if training_procedure == 'kNN':
            model.classifier = nn.Sequential(*[model.classifier[i] for i in range(6)])
        else:
            model.classifier[6] = nn.Linear(4096, num_classes)

    else:
        model = timm.create_model(architecture, pretrained=True, num_classes=num_classes)

    if training_procedure == 'kNN':
        pass

    elif training_procedure == 'endToEnd':
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)

    elif training_procedure == 'linearProbing':
        # Get the state dict
        model_state_dict = model.state_dict()
        # Get the length of the state dict
        state_dict_length = len(model_state_dict)

        state_dict = torch.load(model_path, map_location='cpu')
        len_new_state_dict = len(state_dict)

        print(f"Length {state_dict_length}")

        if architecture == "alexnet":
            if state_dict_length == len_new_state_dict:
                model.load_state_dict(state_dict)
            elif len_new_state_dict == 2:
                model.classifier[6].weight.data = state_dict["weight"]
                model.classifier[6].bias.data = state_dict["bias"]

        elif architecture == "samvit_base_patch16":
            if state_dict_length == len_new_state_dict:
                model.load_state_dict(state_dict)
            elif len_new_state_dict == 2:
                model.head.fc.weight.data = state_dict["weight"]
                model.head.fc.bias.data = state_dict["bias"]

        else:
            if state_dict_length == len_new_state_dict:
                model.load_state_dict(state_dict)
            elif len_new_state_dict == 2:
                model.get_classifier().weight.data = state_dict['weight']
                model.get_classifier().bias.data = state_dict['bias']

    else:
        raise ValueError("Training procedure not supported.")

    # Move the model to the available device
    model = model.to(device)
    model.requires_grad_(False)
    model.eval()

    if training_procedure == 'kNN':
        print("\tCreate the support set ...")
        # Create the support set on the training data
        if architecture == 'alexnet':
            support_set = extract_embeddings_alexnet(model, device, train_loader)

        elif architecture == 'densenet121':
            support_set = extract_embeddings_densenet(model, device, train_loader)

        else:
            support_set = extract_embeddings(model, device, train_loader)

        # Fit the NearestNeighbors model on the support set
        nbrs = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(support_set['embeddings'])

    if task == "multi-label, binary-class":
        prediction = nn.Sigmoid()
    else:
        prediction = nn.Softmax(dim=1)

    # Run the Evaluation
    print(f"\tRun the evaluation ...")
    y_true, y_pred = torch.tensor([]).to(device), torch.tensor([]).to(device)

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            # Map the data to the available device
            images, labels = images.to(device), labels.to(torch.float32).to(device)

            if training_procedure == 'kNN':
                if architecture == 'alexnet':
                    outputs = model(images)

                elif architecture == 'densenet121':
                    outputs = model.forward_features(images)
                    outputs = model.global_pool(outputs)

                else:
                    outputs = model.forward_features(images)
                    outputs = model.forward_head(outputs, pre_logits=True)

                outputs = outputs.reshape(outputs.shape[0], -1)
                outputs = outputs.detach().cpu().numpy()
                outputs = knn_majority_vote(nbrs, outputs, support_set['labels'], task)
                outputs = outputs.to(device)

            else:
                outputs = model(images)
                outputs = prediction(outputs)

            # Store the labels and predictions
            y_true = torch.cat((y_true, deepcopy(labels)), 0)
            y_pred = torch.cat((y_pred, deepcopy(outputs)), 0)

        # Calculate the metrics
        if training_procedure == 'kNN':
            ACC = get_ACC_kNN(y_true.cpu().numpy(), y_pred.cpu().numpy(), task)
            AUC = 0.0  # AUC cannot be calculated for the kNN approach
            BALACC = get_Balanced_ACC_kNN(y_true.cpu().numpy(), y_pred.cpu().numpy(), task)
            Co = get_Cohen_kNN(y_true.cpu().numpy(), y_pred.cpu().numpy(), task)
            Prec = get_Precision_kNN(y_true.cpu().numpy(), y_pred.cpu().numpy(), task)

        else:
            ACC = getACC(y_true.cpu().numpy(), y_pred.cpu().numpy(), task)
            AUC = getAUC(y_true.cpu().numpy(), y_pred.cpu().numpy(), task)
            BALACC = get_Balanced_ACC(y_true.cpu().numpy(), y_pred.cpu().numpy(), task)
            Co = get_Cohen(y_true.cpu().numpy(), y_pred.cpu().numpy(), task)
            Prec = get_Precision(y_true.cpu().numpy(), y_pred.cpu().numpy(), task)

        df_dict = {'dataset_name': dataset_name, 'model_path': model_path, 'training_procedure': training_procedure,
                   'architecture': architecture, 'img_size': img_size, 'ACC': ACC, 'AUC': AUC, 'BALACC': BALACC,
                   'Co': Co, 'Prec': Prec}
        logging_dict_len = [len(df_dict)]
        df = pd.DataFrame(data=df_dict, index=logging_dict_len)
        df.to_csv(save_path, index=False)

    print(f"\tFinished evaluation.")
    # Stop the time
    end_time = time.time()
    hours, minutes, seconds = calculate_passed_time(start_time, end_time)
    print("\tElapsed time for evaluation: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))


if __name__ == '__main__':
    # Read out the command line parameters.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str, help="Path to the configuration file to use.")
    # parser.add_argument("--dataset", required=False, type=str, help="Which dataset to use.")
    # parser.add_argument("--img_size", required=False, type=int, help="Which image size to use.")
    # parser.add_argument("--training_procedure", required=False, type=str, help="Which training procedure to use.")
    # parser.add_argument("--architecture", required=False, type=str, help="Which architecture to use.")
    parser.add_argument("--k", required=False, type=int, help="Number of nearest neighbors to use.")
    # parser.add_argument("--seed", required=False, type=int, help="Which seed was used during training.")

    args = parser.parse_args()
    config_file = args.config_file

    directory = '/mnt/data/benchmark_pretrained_models'
    save_suffix = "test_metrics"  # also suffix for .csv-files containing the metrics

    # works by reading all the model and seed combinations from the existing models in the directory and then performing
    # kNN based on the variations of these models, resolutions and seeds
    training_paradigm = 'endToEnd'  # change to 'endToEnd' or 'linearProbing'
    kNN_true = True

    # Lists all files that contain the training paradigm in their name
    file_names = sorted(
        [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and training_paradigm in f])

    for file_name in file_names[1350:]:
        # Splits the file name by underscores
        parts = file_name.split('_')
        print(parts)

        # Extracts values based on the position of parts
        dataset_name = parts[0]
        img_size = int(parts[1])
        if kNN_true:
            training_procedure = 'kNN'
        else:
            training_procedure = parts[2]
        if parts[3] == "vit" or "samvit" or "eva02" or "efficientnet":
            architecture = '_'.join(parts[3:-2])
        else:
            architecture = parts[3]
        seed_with_s = parts[-2]
        seed = int(seed_with_s[1:])
        save_part = file_name.split('_best.pth')[0]
        if training_procedure == 'kNN':
            save_part = save_part.replace('endToEnd', 'kNN')
        # creating path for saving the .csv-file containing the metrics
        save_path = os.path.join("/mnt/data/medical_benchmark/reevaluation", f"{save_part}_{save_suffix}.csv")
        print(save_path)

        # Displays the extracted information
        print(f"File: {file_name}")
        print(f"Dataset Name: {dataset_name}")
        print(f"Image Size: {img_size}")
        print(f"Training Procedure: {training_procedure}")
        print(f"Architecture: {architecture}")
        print(f"Seed: {seed}")
        print()

        # Loads the parameters and hyperparameters of the configuration file
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Adapts to the command line arguments

        if args.k:
            config['k'] = args.k

        # Seed the training and data loading so both become deterministic
        if architecture == 'alexnet':
            torch.backends.cudnn.benchmark = True  # Enable the benchmark mode in cudnn
            torch.backends.cudnn.deterministic = False  # Disable cudnn to be deterministic
            torch.use_deterministic_algorithms(False)  # Disable only deterministic algorithms

        else:
            torch.backends.cudnn.benchmark = False  # Disable the benchmark mode in cudnn
            torch.backends.cudnn.deterministic = True  # Enable cudnn to be deterministic

            if architecture == 'samvit_base_patch16':
                torch.use_deterministic_algorithms(True, warn_only=True)  # Enable only deterministic algorithms

            else:
                torch.use_deterministic_algorithms(True)  # Enable only deterministic algorithms

        torch.manual_seed(seed)  # Seed the pytorch RNG for all devices (both CPU and CUDA)
        random.seed(seed)
        np.random.seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)

        # Extract the dataset and its metadata
        info = INFO[dataset_name]
        task, in_channel, num_classes = info['task'], info['n_channels'], len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])

        # Create the data transforms and normalize with imagenet statistics
        if architecture == 'alexnet':
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # Use ImageNet statistics
        else:
            m = timm.create_model(architecture, pretrained=True)
            mean, std = m.default_cfg['mean'], m.default_cfg['std']

        total_padding = max(0, 224 - img_size)
        padding_left, padding_top = total_padding // 2, total_padding // 2
        padding_right, padding_bottom = total_padding - padding_left, total_padding - padding_top

        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0,
                           padding_mode='constant')
            # Pad the image to 224x224
        ])

        # Create the datasets
        train_dataset = DataClass(split='train', transform=data_transform, download=False, as_rgb=True,
                                  size=img_size)
        test_dataset = DataClass(split='test', transform=data_transform, download=False, as_rgb=True, size=img_size)

        batch_size = 64
        device = 'cuda:0'

        # Create the dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                  worker_init_fn=seed_worker, generator=g)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                 worker_init_fn=seed_worker, generator=g)

        model_path = f"/mnt/data/benchmark_pretrained_models/{file_name}"
        # Run the evaluation
        evaluate(model_path=model_path, training_procedure=training_procedure, architecture=architecture,
                 img_size=img_size, num_classes=num_classes, device=device, task=task, save_path=save_path,
                 train_loader=train_loader, test_loader=test_loader)
