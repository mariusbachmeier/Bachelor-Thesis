import torch
import time
import torch.nn as nn
import numpy as np
import wandb
from medmnist import INFO
from copy import deepcopy
from timm.optim import AdamW
from tqdm import tqdm
from datetime import datetime
import os
import pandas as pd
from model import MultiNet

from timm.scheduler import CosineLRScheduler
from utils import calculate_passed_time, getACC, getAUC, get_Balanced_ACC, \
    get_Cohen, get_Precision
from backbones import get_backbone


# function to generate a dictionary holding the dataloaders for the 12 specified 2D datasets of medmnist+
def generate_task_dict(dataset_names):
    """
    Generates the dictionary that holds the task as a string given an index which resembles a dataset.

    param dataset_names: iterable which holds strings, each naming a dataset
    """

    print(f"Getting tasks for the {len(dataset_names)} datasets {dataset_names}")
    num_class_dict = {}  # dictionary that holds the number of classes corresponding to the datasets
    task_dict = {}  # dictionary that holds the different tasks corresponding to the datasets
    # fills num_class_dict with the number of output features or classes corresponding to the task of the dataset
    # fills task_dict with the task corresponding to the dataset
    for dataset_name in dataset_names:
        task_string = INFO[dataset_name]['task']
        task_dict[dataset_name] = task_string
        num_classes = len(INFO[dataset_name]['label'])
        num_class_dict[dataset_name] = num_classes
        print(f"The dataset {dataset_name} is a {task_string} classification task and has {num_classes} output classes")
    return num_class_dict, task_dict


def validate_dicts(dataset_names, num_classes_dict: dict, task_dict: dict):
    """
    Performs a few simple checks like whether the dictionaries with the datasets and tasks match in length and whether
    the task matches with the number of classes of the classification layer.

    param dataset_names: Iterable which holds strings, each naming a dataset.
    param num_classes_dict: Holds a dictionary with the number of classes, also number of output features, of the last
    layer, indexed by the dataset name.
    param task_dict: Holds a dictionary with the strings describing the task, indexed by the dataset name.
    """

    # check whether length of dictionaries matches
    if len(num_classes_dict) != len(task_dict):
        print(f"num_classes_dict and task_dict are of unequal length"
              f"num_classes_dict: \n {num_classes_dict}"
              f"task_dict: \n {task_dict}"
              )
        raise ValueError("num_classes_dict and task_dict are of unequal length")
    # check whether task is suited for the number of classes
    for dataset_name in dataset_names:
        current_task = task_dict[dataset_name]
        current_num_classes = num_classes_dict[dataset_name]
        condition_1 = (current_task == 'binary-class' and current_num_classes != 2)
        condition_2 = (current_task == 'multi-class' and current_num_classes <= 2)
        condition_3 = (current_task == 'multi-label, binary-class' and current_num_classes <= 2)
        if condition_1 or condition_2 or condition_3:
            print(f"The number of classes do not match the given task"
                  f"Dataset: {dataset_name}, Task: {task_dict[dataset_name]} "
                  f"Number of Classes: {num_classes_dict[dataset_name]}")
            raise ValueError("The number of classes do not match the given task")


def get_optimizer(optimizer: str, learning_rate: float, model_params):
    """
    Returns the optimizer that's being used.

    param optimizer: holds a string specifying which optimizer is to be used
    param learning rate: the learning rate being used for training
    param model_params: the model parameters
    """
    match optimizer:
        case 'AdamW':
            return AdamW(model_params, lr=learning_rate)


def multi_head_multi_domain_training(config: dict, loader_dict):
    """
    Train a model according to the multi-domain multi-task training paradigm.

    :param config: Dictionary containing the parameters and hyperparameters.
    :param loader_dict: Holds two dictionaries with a collection of dataloaders inside them, one dictionary for the
    collection of dataloaders for the training split, the other dictionary for the collection of dataloaders for the
    validation split
    """

    # Start code
    start_time = time.time()
    print("\tStart training ...")
    # use wandb to track the training
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="final-runs",

        # track hyperparameters and run metadata
        config={
            'output_path': config['output_path'],
            'learning_rate': config['learning_rate'],
            'use_lr_scheduler': config['use_lr_scheduler'],
            'architecture_name': config['architecture_name'],
            'epochs': config['epochs'],
            'early_stopping_epochs': config['early_stopping_epochs'],
            'img_size': config['img_size'],
            'batch_size': config['batch_size'],
            'optimizer': config['optimizer'],
            'num_workers': config['num_workers'],
            'device': config['device'],
            'seed': config['seed'],
            'weighted_loss': config['weighted_loss'],
            'backbone_pretrained': config['backbone_pretrained'],
            'train_backbone': config['train_backbone']
        }
    )

    # variables for better readability later in code
    learning_rate = config['learning_rate']
    use_lr_scheduler = config['use_lr_scheduler']
    architecture = config['architecture_name']
    backbone_name = config['architecture_name']
    num_epochs = config['epochs']
    early_stopping_epochs = config['early_stopping_epochs']
    img_size = config['img_size']
    batch_size = config['batch_size']
    optimizer_str = config['optimizer']
    num_workers = config['num_workers']
    device = config['device']
    seed = config['seed']
    weighted_loss = config['weighted_loss']
    backbone_pretrained = config['backbone_pretrained']

    # log basic info about this training run on console
    print(f"This training uses the {architecture} architecture.\n"
          f"The backbone is initialized with pretrained weights: {str(backbone_pretrained)}.\n"
          f"Images of size {img_size} x {img_size} will be used.\n"
          f"Training will run for {num_epochs} epochs with {early_stopping_epochs} epochs for early stopping.\n"
          f"Training will be done using a learning rate of {learning_rate} and a batch size of {batch_size}.\n"
          f"The loss will be weighted: {str(weighted_loss)}.\n"
          f"The optimizer will be {optimizer_str} and the number of workers {num_workers}.\n"
          f"The device {device} will be used.\n"
          f"The random seed is {seed}.\n")

    # hyperparameter dict for saving information about hyperparameters and some general information
    hyper_dict = {'architecture': backbone_name, 'optimizer': optimizer_str, 'use_lr_scheduler': str(use_lr_scheduler),
                  'img_size': img_size, 'learning_rate': learning_rate, 'batch_size': batch_size, 'epochs': num_epochs,
                  'early_stopping-epochs': early_stopping_epochs, 'num_workers': num_workers, 'seed': seed,
                  'device': device, 'weighted_loss': str(weighted_loss)}
    hyper_dict_len = [len(hyper_dict)]  # needed because each key has a scalar as value
    hyper_df = pd.DataFrame(data=hyper_dict, index=hyper_dict_len)  # declaring index to deal with scalars as values
    # variable for dataframe which will be saved as .csv-file later
    df = pd.DataFrame()

    # creates folder in which models will be saved, named after the current timestamp
    base_path = config['output_path']  # directory into which the new folder for the models will be saved
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  # save current time in format YYYY-mm-dd-HH-MM-SS
    save_path = base_path + current_time  # path where models will be saved later on
    print(f"The save path is: {save_path}")
    os.makedirs(save_path, exist_ok=True)

    # initialise backbone
    backbone, num_features = get_backbone(backbone_name=backbone_name, architecture=architecture, num_classes=1000,
                                          pretrained=backbone_pretrained)
    # check whether there exist as many train loaders as validation loaders
    if list(loader_dict['train_loader_dict'].keys()) == list(loader_dict['val_loader_dict'].keys()):
        dataset_names = list(loader_dict['train_loader_dict'].keys())  # crucial list for indexing, holds dataset names
    else:
        raise ValueError(
            "The Train Loader dictionary and the Validation Loader dictionary contain an unequal amount of loaders")

    # initialize dictionaries, num_class_dict holds the different number of classes the final output layer needs
    # for each dataset, task_dict holds all the tasks corresponding to the datasets, all indexable by means of the
    # dataset name
    num_classes_dict, task_dict = generate_task_dict(dataset_names=dataset_names)
    validate_dicts(dataset_names=dataset_names, num_classes_dict=num_classes_dict, task_dict=task_dict)

    # initialize model suited for the multi-domain multi-task training paradigm
    model = MultiNet(backbone=backbone, num_features=num_features, num_class_dict=num_classes_dict, device=device)
    # Move the model to the available device
    model = model.to(device)

    # Create the optimizer
    print("\tCreate the optimizer")
    optimizer = get_optimizer(optimizer=optimizer_str, learning_rate=learning_rate, model_params=model.parameters())

    # Get lengths of training and validation dataloaders
    loader_len_train = {}  # dictionary holding the lengths of the training dataloaders for each dataset
    loader_len_val = {}  # dictionary holding the lengths of the validation dataloaders for each dataset
    sum_loader_length_train = 0  # holds total length of all the training dataloaders in a given split combined
    sum_loader_length_val = 0  # holds total length of all the validation dataloaders in a given split combined
    for dataset_name, train_loader in loader_dict['train_loader_dict'].items():
        loader_len_train[dataset_name] = len(train_loader)
        sum_loader_length_train += len(train_loader)  # sums up the lengths of all the train dataloaders to obtain the
        # total number of batches inside them
    for dataset_name, val_loader in loader_dict['val_loader_dict'].items():
        loader_len_val[dataset_name] = len(val_loader)
        sum_loader_length_val += len(val_loader)  # sums up the lengths of all the validation dataloaders to obtain the
        # total number of batches inside them

    # Compute initial probabilities of picking dataloader to get a batch from
    probability_dict, train_weights_dict, val_weights_dict = {}, {}, {}
    # weighted such that loss of smallest dataset will receive weight = 1, others fractions of that
    min_len_trainloader = min(loader_len_train, key=loader_len_train.get)
    min_len_valloader = min(loader_len_val, key=loader_len_val.get)
    for dataset_name in dataset_names:
        probability_dict[dataset_name] = loader_len_train[dataset_name] / sum_loader_length_train
        train_weights_dict[dataset_name] = loader_len_train[min_len_trainloader] / loader_len_train[dataset_name]
        val_weights_dict[dataset_name] = loader_len_val[min_len_valloader] / loader_len_val[dataset_name]

    if use_lr_scheduler:
        scheduler = CosineLRScheduler(optimizer, t_initial=num_epochs, cycle_limit=1, t_in_epochs=True)

    # Create variables to store the best performing model
    print("\tInitialize helper variables ...")
    best_loss = np.inf
    best_epoch = 0
    best_model = deepcopy(model)
    best_optimizer = deepcopy(optimizer)
    if use_lr_scheduler:
        best_scheduler = deepcopy(scheduler)
    epochs_no_improve = 0  # Counter for epochs without improvement

    running_epoch = 0

    # Training loop
    print(f"\tRun the training for {config['epochs']} epochs ...")
    for epoch in range(num_epochs):
        running_epoch += 1
        start_time_epoch = time.time()  # Stop the time
        print(f"\t\tEpoch {epoch + 1} of {config['epochs']}:")

        # reset length dictionary and thus the probabilities each epoch
        len_dict = deepcopy(loader_len_train)  # holds a dictionary with the lengths of each training dataloader
        running_probability_dict = deepcopy(probability_dict)  # adaptable dictionary holding probabilities with which a
        # batch should be selected, reset each epoch
        print(f"Probability dictionary of epoch {running_epoch}: {running_probability_dict}")

        # reset batch-wise training metrics for an epoch each epoch
        train_acc_list = []

        # Training
        print(f"\t\t\t Train:")
        model.train()
        # reset training loss each epoch
        train_loss = 0

        # initialize iterables dictionary holding the dataloaders as iterables
        iter_loader_dict = {}
        for dataset_name in dataset_names:
            iter_loader_dict[dataset_name] = iter(loader_dict['train_loader_dict'][dataset_name])

        running_loader_length_sum = deepcopy(sum_loader_length_train)

        # reset dictionaries used for computing metrics each epoch
        y_true_train_dict, y_pred_train_dict = {}, {}
        dataset_train_losses, dataset_val_losses = {}, {}

        # some initializations enabling later computations
        for dataset_name in dataset_names:
            y_true_train_dict[dataset_name] = torch.tensor([]).to(device)
            y_pred_train_dict[dataset_name] = torch.tensor([]).to(device)
            dataset_train_losses[dataset_name] = []
            dataset_val_losses[dataset_name] = []

        for i in range(sum_loader_length_train):
            # get probabilities from dictionary as list
            prob_list = list(running_probability_dict.values())
            # choose random dataset according to weighted probabilities
            random_dataset = np.random.choice(dataset_names, p=prob_list)
            # get batch from dataset which was randomly chosen
            images, labels = next(iter_loader_dict[random_dataset])

            # adjust probabilities for later use
            if len_dict[random_dataset] > 0:
                len_dict[random_dataset] -= 1
            else:
                raise ValueError("length and probability of dataloader can't be less than 0")

            # update probability dictionary
            running_loader_length_sum -= 1
            # would become zero and thus cause a division by zero at the very last batch, no decrement needed at the
            # very end as next epoch will begin and probabilities will be reset anyway
            if running_loader_length_sum >= 1:
                for dataset_name in dataset_names:
                    running_probability_dict[dataset_name] = len_dict[dataset_name] / running_loader_length_sum

            # Map the data to the available device
            images = images.to(device)

            # set loss functions (has to be done each iteration, so per batch, as task might change)
            if task_dict[random_dataset] == "multi-label, binary-class":
                criterion = nn.BCEWithLogitsLoss().to(device)
                prediction = nn.Sigmoid()
                labels = labels.to(torch.float32).to(device)
            else:
                criterion = nn.CrossEntropyLoss().to(device)
                prediction = nn.Softmax(dim=1)
                labels = torch.squeeze(labels, 1).long().to(device)

            # Perform entire training step including forward pass, backward pass and optimization
            loss, outputs_detached = model.train_step(images=images, labels=labels, dataset_name=random_dataset,
                                                      optimizer=optimizer, criterion=criterion,
                                                      weighted_loss=weighted_loss, weights_dict=train_weights_dict)

            train_loss += loss

            dataset_train_losses[random_dataset].append(loss)

            if task_dict[random_dataset] == 'multi-label, binary-class':
                outputs_detached = prediction(outputs_detached).to(device)
            else:
                outputs_detached = prediction(outputs_detached).to(device)
                labels = labels.float().resize_(len(labels), 1)

            # dictionaries holding the predicted and the true labels according to their respective datasets
            y_true_train_dict[random_dataset] = torch.cat((y_true_train_dict[random_dataset], deepcopy(labels)), 0)
            y_pred_train_dict[random_dataset] = torch.cat(
                (y_pred_train_dict[random_dataset], deepcopy(outputs_detached)), 0)

            # get metrics for single batch, as it relies on the task which might change each batch
            train_acc_list.append(
                getACC(labels.cpu().numpy(), outputs_detached.cpu().numpy(), task_dict[random_dataset]))

        # average all the batch-wise accuracies to obtain "global" accuracy per epoch
        ACC_Train_batch = np.mean(train_acc_list)

        # The following dictionaries hold the metrics for each dataset individually instead of for the union of datasets
        train_acc_dict, train_auc_dict, train_balacc_dict, train_co_dict, train_prec_dict, train_loss_dict = ({}, {},
                                                                                                              {}, {},
                                                                                                              {}, {})
        for dataset_name in dataset_names:
            # metrics are computed for the entirety of batches of the dataset, thus resembling the metric per dataset
            train_acc_dict[dataset_name] = getACC(y_true_train_dict[dataset_name].cpu().numpy(),
                                                  y_pred_train_dict[dataset_name].cpu().numpy(),
                                                  task_dict[dataset_name])
            train_auc_dict[dataset_name] = getAUC(y_true_train_dict[dataset_name].cpu().numpy(),
                                                  y_pred_train_dict[dataset_name].cpu().numpy(),
                                                  task_dict[dataset_name])
            train_balacc_dict[dataset_name] = get_Balanced_ACC(y_true_train_dict[dataset_name].cpu().numpy(),
                                                               y_pred_train_dict[dataset_name].cpu().numpy(),
                                                               task_dict[dataset_name])
            train_co_dict[dataset_name] = get_Cohen(y_true_train_dict[dataset_name].cpu().numpy(),
                                                    y_pred_train_dict[dataset_name].cpu().numpy(),
                                                    task_dict[dataset_name])
            train_prec_dict[dataset_name] = get_Precision(y_true_train_dict[dataset_name].cpu().numpy(),
                                                          y_pred_train_dict[dataset_name].cpu().numpy(),
                                                          task_dict[dataset_name])
            train_loss_dict[dataset_name] = np.mean(dataset_train_losses[dataset_name])

        # calculate metrics such that each dataset has the same amount of "weight"
        ACC_Train = np.mean(np.fromiter(train_acc_dict.values(), dtype=float))
        AUC_Train = np.mean(np.fromiter(train_auc_dict.values(), dtype=float))
        Bal_Acc_Train = np.mean(np.fromiter(train_balacc_dict.values(), dtype=float))
        Co_Train = np.mean(np.fromiter(train_co_dict.values(), dtype=float))
        Prec_Train = np.mean(np.fromiter(train_prec_dict.values(), dtype=float))

        # Update the learning rate each epoch
        if use_lr_scheduler:
            scheduler.step(epoch=epoch)

        # Evaluation
        print(f"\t\t\t Evaluate:")
        model.eval()
        # reset validation loss each epoch
        val_loss = 0
        # reset dictionaries holding the ground truth and the predicted labels each epoch
        y_true_val_dict, y_pred_val_dict = {}, {}
        for dataset_name in dataset_names:
            y_true_val_dict[dataset_name] = torch.tensor([]).to(device)
            y_pred_val_dict[dataset_name] = torch.tensor([]).to(device)

        # reset validation metrics each epoch
        val_acc_list, val_auc_list, val_balacc_list, val_co_list, val_prec_list = [], [], [], [], []

        with torch.no_grad():
            # iterating over all validation dataloaders
            for dataset_name, single_val_loader in loader_dict['val_loader_dict'].items():
                if task_dict[dataset_name] == "multi-label, binary-class":
                    criterion = nn.BCEWithLogitsLoss().to(device)
                    prediction = nn.Sigmoid()
                else:
                    criterion = nn.CrossEntropyLoss().to(device)
                    prediction = nn.Softmax(dim=1)

                # iterating over all the batches of a single dataloader
                for images, labels in tqdm(single_val_loader):
                    # Map the data to the available device
                    images = images.to(device)
                    outputs = model(x=images, dataset_name=dataset_name)

                    # Run the forward pass
                    if task_dict[dataset_name] == 'multi-label, binary-class':
                        labels = labels.to(torch.float32).to(device)
                        loss = criterion(outputs, labels)
                        outputs = prediction(outputs).to(device)

                    else:
                        labels = torch.squeeze(labels, 1).long().to(device)
                        loss = criterion(outputs, labels)
                        outputs = prediction(outputs).to(device)
                        labels = labels.float().resize_(len(labels), 1)

                    # weight loss for consistent comparison of training and validation metric
                    if weighted_loss:
                        loss = val_weights_dict[dataset_name] * loss

                    # Store the current loss
                    val_loss += loss.item()

                    dataset_val_losses[dataset_name].append(loss.item())

                    # Store the labels and predictions
                    y_true_val_dict[dataset_name] = torch.cat((y_true_val_dict[dataset_name], deepcopy(labels)),
                                                              0)
                    y_pred_val_dict[dataset_name] = torch.cat((y_pred_val_dict[dataset_name], deepcopy(outputs)), 0)

                # average all the batch-wise accuracies to obtain "global" accuracy per epoch per dataset
                val_acc_list.append(
                    getACC(y_true_val_dict[dataset_name].cpu().numpy(),
                           y_pred_val_dict[dataset_name].cpu().numpy(), task_dict[dataset_name]))

        ACC_batch = np.mean(val_acc_list)

        # The following dictionaries hold the metrics for each dataset individually instead of for the union of datasets
        val_acc_dict, val_auc_dict, val_balacc_dict, val_co_dict, val_prec_dict, val_loss_dict = {}, {}, {}, {}, {}, {}
        for dataset_name in dataset_names:
            val_acc_dict[dataset_name] = getACC(y_true_val_dict[dataset_name].cpu().numpy(),
                                                y_pred_val_dict[dataset_name].cpu().numpy(),
                                                task_dict[dataset_name])
            val_auc_dict[dataset_name] = getAUC(y_true_val_dict[dataset_name].cpu().numpy(),
                                                y_pred_val_dict[dataset_name].cpu().numpy(),
                                                task_dict[dataset_name])
            val_balacc_dict[dataset_name] = get_Balanced_ACC(y_true_val_dict[dataset_name].cpu().numpy(),
                                                             y_pred_val_dict[dataset_name].cpu().numpy(),
                                                             task_dict[dataset_name])
            val_co_dict[dataset_name] = get_Cohen(y_true_val_dict[dataset_name].cpu().numpy(),
                                                  y_pred_val_dict[dataset_name].cpu().numpy(),
                                                  task_dict[dataset_name])
            val_prec_dict[dataset_name] = get_Precision(y_true_val_dict[dataset_name].cpu().numpy(),
                                                        y_pred_val_dict[dataset_name].cpu().numpy(),
                                                        task_dict[dataset_name])
            val_loss_dict[dataset_name] = np.mean(dataset_val_losses[dataset_name])

        # calculate metrics such that each dataset has the same amount of "weight"
        ACC = np.mean(np.fromiter(val_acc_dict.values(), dtype=float))
        AUC = np.mean(np.fromiter(val_auc_dict.values(), dtype=float))
        Bal_Acc = np.mean(np.fromiter(val_balacc_dict.values(), dtype=float))
        Co = np.mean(np.fromiter(val_co_dict.values(), dtype=float))
        Prec = np.mean(np.fromiter(val_prec_dict.values(), dtype=float))

        # Print the loss values and send them to wandb
        train_loss /= sum_loader_length_train
        val_loss /= sum_loader_length_val
        print(f"\t\t\tTrain Loss: {train_loss}")
        print(f"\t\t\tVal Loss: {val_loss}")
        print(f"\t\t\tTrain ACC: {ACC_Train}")
        print(f"\t\t\tVal ACC: {ACC}")

        # Creating dictionary holding both the metrics for the union of datasets as well as the metrics for each
        # individual dataset
        logging_dict = {"epochs": epoch, "train_loss": train_loss, "val_loss": val_loss, "accuracy_train": ACC_Train,
                        "accuracy_val": ACC, "accuracy_train_batch": ACC_Train_batch, "accuracy_val_batch": ACC_batch,
                        "auc_train": AUC_Train, "auc_val": AUC, "balacc_train": Bal_Acc_Train,
                        "balacc_val": Bal_Acc, "co_train": Co_Train, "co_val": Co, "prec_train": Prec_Train,
                        "prec_val": Prec
                        }

        for dataset_name in dataset_names:
            logging_dict[dataset_name + '_train_acc'] = train_acc_dict[dataset_name]
            logging_dict[dataset_name + '_train_AUC'] = train_auc_dict[dataset_name]
            logging_dict[dataset_name + '_train_balacc'] = train_balacc_dict[dataset_name]
            logging_dict[dataset_name + '_train_co'] = train_co_dict[dataset_name]
            logging_dict[dataset_name + '_train_prec'] = train_prec_dict[dataset_name]
            logging_dict[dataset_name + '_train_loss'] = train_loss_dict[dataset_name]
            logging_dict[dataset_name + '_val_acc'] = val_acc_dict[dataset_name]
            logging_dict[dataset_name + '_val_AUC'] = val_auc_dict[dataset_name]
            logging_dict[dataset_name + '_val_balacc'] = val_balacc_dict[dataset_name]
            logging_dict[dataset_name + '_val_co'] = val_co_dict[dataset_name]
            logging_dict[dataset_name + '_val_prec'] = val_prec_dict[dataset_name]
            logging_dict[dataset_name + '_val_loss'] = val_loss_dict[dataset_name]

        # logging all the metrics
        wandb.log(logging_dict)

        if df.empty:
            logging_dict_len = [len(logging_dict)]
            df = pd.DataFrame(data=logging_dict, index=logging_dict_len)
        else:
            logging_dict_len = [len(logging_dict.keys())]
            df_concat = pd.DataFrame(data=logging_dict, index=logging_dict_len)
            df = pd.concat([df, df_concat])

        # Store the current best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_model = deepcopy(model)
            best_optimizer = deepcopy(optimizer)
            best_scheduler = deepcopy(scheduler)
            best_metrics = deepcopy(df)
            epochs_no_improve = 0  # Reset the counter
        else:
            epochs_no_improve += 1  # Increment the counter

        print(f"\t\t\tCurrent best Validation Loss: {best_loss}")
        print(f"\t\t\tCurrent best Epoch: {best_epoch}")

        # Check for early stopping
        if epochs_no_improve == early_stopping_epochs:
            print("\tEarly stopping!")
            break

        # Saving latest model
        latest_dict = {'model_parameters': model.state_dict(), 'optimizer': optimizer.state_dict(),
                       'scheduler_state_dict': scheduler.state_dict(), 'epochs': running_epoch}
        torch.save(latest_dict, save_path + '/latest_model.pth')
        df.to_csv(save_path + "/latest_metrics.csv", index=False)

        # Stop the time for the epoch
        end_time_epoch = time.time()
        hours_epoch, minutes_epoch, seconds_epoch = calculate_passed_time(start_time_epoch, end_time_epoch)
        print("\t\t\tElapsed time for epoch: {:0>2}:{:0>2}:{:05.2f}".format(hours_epoch, minutes_epoch, seconds_epoch))

    # Reset all parameters to trainable after training is done
    for param in model.parameters():
        param.requires_grad = True

    # Saving the final (and not necessarily best) model
    final_dict = {'model_parameters': model.state_dict(), 'optimizer': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict(),
                  'epochs': running_epoch}
    torch.save(final_dict, save_path + '/final_model.pth')
    # Saving the final Metrics
    df.to_csv(save_path + "/final_metrics.csv", index=False)
    # Saving the Hyperparameters that were used
    hyper_df.to_csv(save_path + "/hyperparameters.csv", index=False)
    # Saving the best model
    best_dict = {'model_parameters': best_model.state_dict(), 'optimizer': best_optimizer.state_dict(),
                 'scheduler_state_dict': best_scheduler.state_dict(), 'epochs': best_epoch}
    torch.save(best_dict, save_path + '/best_model.pth')
    # Saving the best metrics
    best_metrics.to_csv(save_path + "/best_metrics.csv", index=False)
