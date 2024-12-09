import argparse
import math
from medmnist import INFO
import torch
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms
import timm
import numpy as np
import random
import yaml
import time
import pandas as pd
from copy import deepcopy
from utils import seed_worker
from data import generate_datasets, generate_dataloaders
from model import MultiNet
from backbones import get_backbone
from multi_head_multi_domain_pt import generate_task_dict, validate_dicts
from utils import calculate_passed_time, getACC, getAUC, get_Balanced_ACC, \
    get_Cohen, get_Precision
import sys
import ast


def validate_metrics_dicts(dict_1: dict, dict_2: dict):
    """
        Validates that the metrics computed from the .csv match the metrics computed during evaluation

        param dict_1: holds dictionary with metrics obtained during training
        param dict_1: holds dictionary with metrics obtained from .csv-file with predictions and ground truths
        """
    # First, ensure both dictionaries have the same keys
    if dict_1.keys() != dict_2.keys():
        return False

    # Now, iterate over the keys and compare values
    for key in dict_1:
        val_1 = dict_1[key]
        val_2 = dict_2[key]

        # If both values are NaN, consider them equal and move on
        if isinstance(val_1, float) and isinstance(val_2, float) and math.isnan(val_1) and math.isnan(val_2):
            continue

        # Otherwise, check for regular equality
        if val_1 != val_2:
            return False

    return True


def test():
    # specifying the .yaml file which is used
    sys.argv = ['test.py', '--config_file=mm-pt_config.yaml',
                '--path=<path to folder in which (best) trained model was stored>']

    # parsing the path to the specified -yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, type=str, help="Path to the configuration file to use.")
    parser.add_argument("--path", required=True, type=str, help="Path to the model directory.")

    # Tracking the time it takes for this test script to run
    start_time = time.time()
    print("\tStart test script ...")

    args = parser.parse_args()
    config_file = args.config_file
    path = args.path  # Use the provided path argument

    # Load the parameters and hyperparameters of the yaml configuration file
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # for reproducibility
    torch.backends.cudnn.benchmark = False  # Disable the benchmark mode in cudnn
    torch.backends.cudnn.deterministic = True  # Enable cudnn to be deterministic
    torch.use_deterministic_algorithms(True)  # Enable only deterministic algorithms
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])  # Seed the pytorch RNG for all devices (both CPU and CUDA)
    np.random.seed(config['seed'])
    g = torch.Generator()
    g.manual_seed(config['seed'])

    device = config['device']
    img_size = config['img_size']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    architecture_name = config['architecture_name']
    backbone_name = config['architecture_name']
    backbone_pretrained = config['backbone_pretrained']

    # Image Padding with zeros, needed if img_size != 224
    total_padding = max(0, 224 - img_size)
    padding_left, padding_top = total_padding // 2, total_padding // 2
    padding_right, padding_bottom = total_padding - padding_left, total_padding - padding_top

    # Transformations
    m = timm.create_model(architecture_name, pretrained=True)
    mean, std = m.default_cfg['mean'], m.default_cfg['std']
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=0,
                       padding_mode='constant')  # Zero-Pad the image to 224x224
    ])

    # Create list of dataset names
    dataset_names = list(INFO)[:12]  # holds the names of the 12 medmnist+ datasets as a list of strings

    # generate a dictionary holding the test datasets
    test_ds_dict = generate_datasets(dataset_names=dataset_names, split="test", img_size=img_size,
                                     transformations=data_transform)

    # generate a dictionary holding the test dataloaders
    test_loader_dict = generate_dataloaders(dataset_dict=test_ds_dict, num_workers=num_workers,
                                            worker_seed=seed_worker, gen=g, shuffle=False, batch_size=batch_size,
                                            split="test")

    # loading the model
    save_dict = torch.load(f=path + r'\best_model.pth')
    model_state_dict = save_dict['model_parameters']

    # initialize backbone
    backbone, num_features = get_backbone(backbone_name=backbone_name, architecture=architecture_name, num_classes=1000,
                                          pretrained=backbone_pretrained)
    num_classes_dict, task_dict = generate_task_dict(dataset_names=dataset_names)
    validate_dicts(dataset_names=dataset_names, num_classes_dict=num_classes_dict, task_dict=task_dict)

    # initialize the model
    model = MultiNet(backbone=backbone, num_features=num_features, num_class_dict=num_classes_dict, device=device)
    # Move the model to the available device
    model = model.to(device)
    model.load_state_dict(model_state_dict)

    # initialize dataframe that holds the test results
    results_df = pd.DataFrame(columns=['dataset_name', 'prediction', 'truth', 'probability'])
    # initialize dictionary that holds the test metrics
    metrics_dict = {}

    # Testing loop
    print(f"\t\t\t Test:")
    model.eval()

    # for calculating metrics, holds ground truths and predictions
    y_true_test_dict, y_pred_test_dict = {}, {}
    for dataset_name in dataset_names:
        y_true_test_dict[dataset_name] = torch.tensor([]).to(device)
        y_pred_test_dict[dataset_name] = torch.tensor([]).to(device)

    # hold metrics
    test_acc_list, test_auc_list, test_balacc_list, test_co_list, test_prec_list = [], [], [], [], []

    # Test Evaluation
    with torch.no_grad():
        # iterating over all test dataloaders
        for dataset_name, single_test_loader in test_loader_dict.items():
            if task_dict[dataset_name] == "multi-label, binary-class":
                prediction = nn.Sigmoid()
            else:
                prediction = nn.Softmax(dim=1)

            # iterating over all the batches of a single dataloader
            for images, labels in tqdm(single_test_loader):
                # Map the data to the available device
                images = images.to(device)
                outputs = model(x=images, dataset_name=dataset_name)

                # Run the forward pass
                if task_dict[dataset_name] == 'multi-label, binary-class':
                    labels = labels.to(torch.float32).to(device)
                    outputs = prediction(outputs).to(device)
                    # obtain predicted labels, using 0.5 as threshold due to binary classification
                    predictions = (outputs > 0.5).int()

                else:
                    labels = torch.squeeze(labels, 1).long().to(device)
                    outputs = prediction(outputs).to(device)
                    labels = labels.float().resize_(len(labels), 1)
                    # obtain predicted labels, using argmax for the label with the highest probability
                    predictions = torch.argmax(outputs, dim=1)

                # Store the labels and predictions
                y_true_test_dict[dataset_name] = torch.cat((y_true_test_dict[dataset_name], deepcopy(labels)),
                                                           0)
                y_pred_test_dict[dataset_name] = torch.cat((y_pred_test_dict[dataset_name], deepcopy(outputs)), 0)

                # use list to append individual labels and predictions as pandas append to df would introduce a
                # lot of computational overhead
                batch_results = []
                for i in range(
                        len(labels)):  # len(labels) as the last batch might not have full length equal the batch size
                    # append results of the batch
                    batch_results.append({
                        'dataset_name': dataset_name,
                        'prediction': predictions[i].cpu().numpy().tolist(),
                        'truth': labels[i].cpu().numpy().tolist(),
                        'probability': outputs[i].cpu().numpy().tolist()})

                # append to df after iterating over batch
                results_df = pd.concat([results_df, pd.DataFrame(batch_results)], ignore_index=True)

            # average all the batch-wise accuracies to obtain "global" accuracy per epoch per dataset
            test_acc_list.append(
                getACC(y_true_test_dict[dataset_name].cpu().numpy(),
                       y_pred_test_dict[dataset_name].cpu().numpy(), task_dict[dataset_name]))

    # accuracy computed on individual batches
    ACC_batch = np.mean(test_acc_list)

    # The following dictionaries hold the metrics for each dataset individually instead of ust for the union of datasets
    test_dict = {}
    for dataset_name in dataset_names:
        test_dict[dataset_name + '_test_acc'] = getACC(y_true_test_dict[dataset_name].cpu().numpy(),
                                                       y_pred_test_dict[dataset_name].cpu().numpy(),
                                                       task_dict[dataset_name])
        test_dict[dataset_name + '_test_AUC'] = getAUC(y_true_test_dict[dataset_name].cpu().numpy(),
                                                       y_pred_test_dict[dataset_name].cpu().numpy(),
                                                       task_dict[dataset_name])
        test_dict[dataset_name + '_test_balacc'] = get_Balanced_ACC(y_true_test_dict[dataset_name].cpu().numpy(),
                                                                    y_pred_test_dict[dataset_name].cpu().numpy(),
                                                                    task_dict[dataset_name])
        test_dict[dataset_name + '_test_co'] = get_Cohen(y_true_test_dict[dataset_name].cpu().numpy(),
                                                         y_pred_test_dict[dataset_name].cpu().numpy(),
                                                         task_dict[dataset_name])
        test_dict[dataset_name + '_test_prec'] = get_Precision(y_true_test_dict[dataset_name].cpu().numpy(),
                                                               y_pred_test_dict[dataset_name].cpu().numpy(),
                                                               task_dict[dataset_name])

    # store the values for all the datasets of each metric in an individual list
    acc_initial = [value for key, value in test_dict.items() if '_test_acc' in key]
    auc_initial = [value for key, value in test_dict.items() if '_test_AUC' in key]
    balanced_acc_initial = [value for key, value in test_dict.items() if '_test_balacc' in key]
    co_initial = [value for key, value in test_dict.items() if '_test_co' in key]
    precision_initial = [value for key, value in test_dict.items() if '_test_prec' in key]

    # calculate metrics such that each dataset has the same amount of "weight"
    ACC = np.mean(np.fromiter(acc_initial, dtype=float))
    AUC = np.mean(np.fromiter(auc_initial, dtype=float))
    Bal_Acc = np.mean(np.fromiter(balanced_acc_initial, dtype=float))
    Co = np.mean(np.fromiter(co_initial, dtype=float))
    Prec = np.mean(np.fromiter(precision_initial, dtype=float))

    test_dict['accuracy_test'] = ACC
    test_dict['auc_test'] = AUC
    test_dict['balacc_test'] = Bal_Acc
    test_dict['co_test'] = Co
    test_dict['prec_test'] = Prec

    print(f"\t\t\tTest ACC: {ACC}")
    print(f"\t\t\tBatch-wise Test ACC: {ACC_batch}")
    print(f"\t\t\tTest AUC: {AUC}")
    print(f"\t\t\tTest Balanced Accuracy: {Bal_Acc}")
    print(f"\t\t\tTest Cohen's Kappa: {Co}")
    print(f"\t\t\tTest Precision: {Prec}")

    results_path = "/test_results.csv"
    # contains the predicted labels and the ground truth labels as well as their probabilities
    results_df.to_csv(path + results_path, index=False)

    # Read the CSV file
    results_df = pd.read_csv(path + results_path)

    # Convert string representations back to lists
    def string_to_list(s):
        return ast.literal_eval(s)

    for col in ['truth', 'probability', 'prediction']:
        results_df[col] = results_df[col].apply(string_to_list)

    # Initialize a dictionary to store metrics computed from the CSV
    metrics_from_csv = {}

    # Iterate over each dataset
    for dataset_name in dataset_names:
        # Filter the Dataframe for the current dataset
        df_dataset = results_df[results_df['dataset_name'] == dataset_name]

        # Extract truths, probabilities, and predictions
        truths = df_dataset['truth'].tolist()
        probabilities = df_dataset['probability'].tolist()

        task = task_dict[dataset_name]

        if task == 'multi-label, binary-class':
            # For multi-label tasks, truths and probabilities are lists of lists
            y_true = np.array(truths)
            y_score = np.array(probabilities)
        else:
            # For multi-class tasks, truths and probabilities need special handling
            y_true = np.array([t[0] if isinstance(t, list) else t for t in truths]).astype(int)
            y_score = np.array(probabilities)

        # Store the metrics
        metrics_from_csv[dataset_name + '_test_acc'] = getACC(y_true, y_score, task)
        metrics_from_csv[dataset_name + '_test_AUC'] = getAUC(y_true, y_score, task)
        metrics_from_csv[dataset_name + '_test_balacc'] = get_Balanced_ACC(y_true, y_score, task)
        metrics_from_csv[dataset_name + '_test_co'] = get_Cohen(y_true, y_score, task)
        metrics_from_csv[dataset_name + '_test_prec'] = get_Precision(y_true, y_score, task)

    acc_csv = [value for key, value in metrics_from_csv.items() if '_test_acc' in key]
    auc_csv = [value for key, value in metrics_from_csv.items() if '_test_AUC' in key]
    balanced_acc_csv = [value for key, value in metrics_from_csv.items() if '_test_balacc' in key]
    co_csv = [value for key, value in metrics_from_csv.items() if '_test_co' in key]
    precision_csv = [value for key, value in metrics_from_csv.items() if '_test_prec' in key]

    # Compute the average metrics, 'weighting' each dataset equally
    average_acc = np.mean(acc_csv)
    average_auc = np.mean(auc_csv)
    average_balanced_acc = np.mean(balanced_acc_csv)
    average_co = np.mean(co_csv)
    average_precision = np.mean(precision_csv)

    # add the metrics which got averaged over each dataset to the dictionary
    metrics_from_csv['accuracy_test'] = average_acc
    metrics_from_csv['auc_test'] = average_auc
    metrics_from_csv['balacc_test'] = average_balanced_acc
    metrics_from_csv['co_test'] = average_co
    metrics_from_csv['prec_test'] = average_precision

    print(f"\t\t\tcsv ACC: {average_acc}")
    print(f"\t\t\tcsv AUC: {average_auc}")
    print(f"\t\t\tcsv Balanced-ACC: {average_balanced_acc}")
    print(f"\t\t\tcsv Cohen's-Kappa': {average_co}")
    print(f"\t\t\tcsv Precision: {average_precision}")

    test_metrics_dict_len = [len(metrics_from_csv)]
    test_metrics_df = pd.DataFrame(data=metrics_from_csv, index=test_metrics_dict_len)

    test_metrics_path = "/test_metrics.csv"
    # contains the predicted labels and the ground truth labels as well as their probabilities
    test_metrics_df.to_csv(path + test_metrics_path, index=False)

    print(test_dict)
    print(metrics_from_csv)
    # validate that the metrics computed from the csv match the metrics computed during evaluation
    if validate_metrics_dicts(dict_1=test_dict, dict_2=metrics_from_csv):
        print("Successfully validated the dictionaries holding the metrics")
    else:
        raise ValueError("Metrics do not match in test_dict and metrics_from_csv")

    # Stop the time for testing
    end_time_epoch = time.time()
    hours_epoch, minutes_epoch, seconds_epoch = calculate_passed_time(start_time, end_time_epoch)
    print("\t\t\tElapsed time for testing: {:0>2}:{:0>2}:{:05.2f}".format(hours_epoch, minutes_epoch, seconds_epoch))


if __name__ == "__main__":
    test()
