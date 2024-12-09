from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO
import torchvision.transforms as transforms
import timm
from utils import seed_worker


def data(config: dict, gen):
    """
        Organizes the dataloader initialisation, sets different necessary variables based on the config, the
        transformations for the dataloaders and calls the functions generate_datasets and generate_dataloaders:

        :param config: Dictionary containing the parameters and hyperparameters.
        :param gen: generator with manual seed
        :return: Returns the dataloaders obtained by calling the functions generate_datasets and generate_dataloaders.
    """

    g = gen
    img_size = config['img_size']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    architecture_name = config['architecture_name']

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

    train_ds_dict = generate_datasets(dataset_names=dataset_names, split="train", img_size=img_size,
                                      transformations=data_transform)
    val_ds_dict = generate_datasets(dataset_names=dataset_names, split="val", img_size=img_size,
                                    transformations=data_transform)

    train_loader_dict = generate_dataloaders(dataset_dict=train_ds_dict, num_workers=num_workers,
                                             worker_seed=seed_worker, gen=g, shuffle=True, batch_size=batch_size,
                                             split="train")
    val_loader_dict = generate_dataloaders(dataset_dict=val_ds_dict, num_workers=num_workers, worker_seed=seed_worker,
                                           gen=g, shuffle=False, batch_size=batch_size, split="val")

    return {'train_loader_dict': train_loader_dict, 'val_loader_dict': val_loader_dict}


def generate_datasets(dataset_names: list[str], split: str, img_size: int, transformations):
    """
        Initializes the datasets and saves them to a dictionary holding the 12 specified 2D datasets of medmnist+
        instantiated with a given split which could be train, validation or test

        :param dataset_names: holds the names of all the datasets as a list of string
        :param split: string that holds the split according to which a dataset should be loaded
        :param img_size: the size of the images that should be loaded
        :param transformations: the transformations that are going to be applied to the images
        :return: dictionary with the datasets 12 specified 2D datasets of MedMNIST+
    """

    print(f"Loading the {len(dataset_names)} datasets {dataset_names} ")
    ds_dict = {}  # dictionary that holds the 12 MEDMNIST+ dataset objects
    # fills dictionary with 12 MEDMNIST+ dataset objects
    for dataset_name in dataset_names:
        DataClass = getattr(medmnist, INFO[dataset_name]['python_class'])
        mnist_dataset = DataClass(split=split, transform=transformations, download=True, as_rgb=True,
                                  size=img_size)
        ds_dict[dataset_name] = mnist_dataset  # assign dataset as 'value' to new key with name of dataset
    return ds_dict


def generate_dataloaders(dataset_dict: dict, num_workers: int, worker_seed, gen, shuffle: bool, batch_size: int,
                         split: str):
    """
        Generates a dictionary holding the dataloaders for the 12 specified 2D datasets of medmnist+

        :param dataset_dict: holds the previously initialized datasets in a dictionary
        :param num_workers: the number of workers specified in the config file
        :param worker_seed: holds the previously initialized datasets
        :param gen: the generator with its manual seed
        :param shuffle: determines whether to shuffle (True) or not (False)
        :param batch_size: the size of the batches which constitute the data in the dataloader
        :param split: denominates the split that's being used, only used for logging
        :return: dictionary with the dataloaders for the 12 specified 2D datasets of MedMNIST+
    """

    dataloader_dict = {}  # dictionary that holds the 12 MEDMNIST+ dataset objects
    # fills dictionary with 12 MEDMNIST+ dataset objects
    for dataset_name, dataset in dataset_dict.items():
        dataloader_dict[dataset_name] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                                   num_workers=num_workers, worker_init_fn=worker_seed, generator=gen)
        print(
            f"Created {split} Dataloader for {dataset_name} datasets containing "
            f"{len(dataloader_dict[dataset_name])} batches")
    return dataloader_dict
