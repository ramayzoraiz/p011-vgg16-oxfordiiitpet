"""
weird workaround!
reason to separate torch.utils.data.Dataset into py file and then import dataset in ipynb file is
to avoid dataloader multiprocessing error(when num_workers>0) in jupyter notebook.
"""
import torch
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import v2
from torch.utils.data import ConcatDataset, random_split, Dataset

#%% simple case 1
def one_hot_encode(x):
    return torch.nn.functional.one_hot(torch.tensor(x), num_classes=37)

def load_data(IMAGE_SIZE):
    """
    load the dataet(PIL image and int label) and transform it to tensor image(uint8[0-255]) -> resize -> range 0 to 1
    and one-hot encode the labels
    split val dataset into 70:30 ratio. Keep 30 to val dataset. Combine 70(% of initial val dataset) into train dataset.
    """
    # Load the dataset; transform PIL to tensor image(uint8[0-255]) and one-hot encode the labels
    # v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_ds = OxfordIIITPet(root='../p008-lenet-cnn-model/pytorch/data', split='trainval', download=False,
                    transform=v2.Compose([v2.ToImage(), v2.Resize((IMAGE_SIZE, IMAGE_SIZE)), v2.ToDtype(torch.float32, scale=True)]),
                    target_transform=v2.Lambda(one_hot_encode) )
    temp_val_ds = OxfordIIITPet(root='../p008-lenet-cnn-model/pytorch/data', split='test', download=False,
                    transform=v2.Compose([v2.ToImage(), v2.Resize((IMAGE_SIZE, IMAGE_SIZE)), v2.ToDtype(torch.float32, scale=True)]),
                    target_transform=v2.Lambda(one_hot_encode) )

    # Define the split ratio for val_ds
    part_train_size = int(0.7 * len(temp_val_ds))
    val_size = len(temp_val_ds) - part_train_size

    # Split the dataset into part_training and val sets
    part_train_ds, val_ds = random_split(temp_val_ds, [part_train_size, val_size], generator=torch.Generator().manual_seed(42))
    # Combine the part_train_ds and train_ds into a single train_ds
    train_ds = ConcatDataset([train_ds, part_train_ds])
    return train_ds, val_ds

#%% normalize case 

class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
def normalize_data(train_ds, val_ds, mean, std):   
    additional_transform = v2.Normalize(mean, std)
    train_ds = TransformDataset(train_ds, transform=additional_transform)
    val_ds = TransformDataset(val_ds, transform=additional_transform)
    return train_ds, val_ds


#%% synthetic data generation case
IMAGE_SIZE = 128
def one_hot_encode2(x):
    return torch.nn.functional.one_hot(torch.tensor(x), num_classes=37)

def load_simple_data():
    """
    load the dataet(PIL image and int label) and transform it to tensor image(uint8[0-255])
    and one-hot encode the labels
    split val dataset into 70:30 ratio. Keep 30 to val dataset. Combine 70(% of initial val dataset) into train dataset.
    """
    # Load the dataset; transform PIL to tensor image(uint8[0-255]) and one-hot encode the labels
    # v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_ds = OxfordIIITPet(root='../p008-lenet-cnn-model/pytorch/data', split='trainval', download=False,
                    transform=v2.ToImage(), target_transform=v2.Lambda(one_hot_encode2) )
    temp_val_ds = OxfordIIITPet(root='../p008-lenet-cnn-model/pytorch/data', split='test', download=False,
                    transform=v2.ToImage(), target_transform=v2.Lambda(one_hot_encode2) )

    # Define the split ratio for val_ds
    part_train_size = int(0.7 * len(temp_val_ds))
    val_size = len(temp_val_ds) - part_train_size

    # Split the dataset into part_training and val sets
    part_train_ds, val_ds = random_split(temp_val_ds, [part_train_size, val_size], generator=torch.Generator().manual_seed(42))
    # Combine the part_train_ds and train_ds into a single train_ds
    train_ds = ConcatDataset([train_ds, part_train_ds])
    return train_ds, val_ds

# Define custom dataset to apply transformations
class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
"""
dataset(uint8) \
    |---> (resize) (rescale) ---> org \
    |---> (central_crop) (resize) (flip_left_right) (recale)---> temp1 \
    |---> (top_left_crop) (resize) (random_hue) (random_flip_left_right) (rescale)---> temp2 \
    |---> (top_right_crop) (resize) (random_brightness) (random_flip_left_right) (rescale)---> temp3 \
    |---> (bottom_left_crop) (resize) (random_saturation) (random_flip_left_right) (rescale)---> temp4 \
    |---> (bottom_right_crop) (resize) (random_contrast) (random_flip_left_right) (rescale)---> temp5 \
    org + temp1 + temp2 + temp3 + temp4 + temp5 ---> train_ds 
"""
# Define transformation functions
def resize(img):
    transform = transform=v2.Compose([v2.Resize((IMAGE_SIZE, IMAGE_SIZE)), v2.ToDtype(torch.float32, scale=True)])
    img = transform(img)
    return img

def central_crop(img):
    transform = v2.Compose([
        v2.CenterCrop(int(IMAGE_SIZE * 0.85)),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        v2.RandomHorizontalFlip(p=1.0),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize((0.5,), (0.5,))
    ])
    img = transform(img)
    return img

def top_left_crop(img):
    h = img.shape[1]
    w = img.shape[2]
    cut = 0.86
    img = img[:,:int(h*cut),:int(w*cut)]
    transform = v2.Compose([
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        v2.ColorJitter(hue=0.45),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize((0.5,), (0.5,))
    ])
    img = transform(img)
    return img

def top_right_crop(img):
    h = img.shape[1]
    w = img.shape[2]
    cut = 0.86
    img = img[:,:int(h*cut),int(w*(1-cut)):]
    transform = v2.Compose([
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        v2.ColorJitter(brightness=0.3),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize((0.5,), (0.5,))
    ])
    img = transform(img)
    return img

def bottom_left_crop(img):
    h = img.shape[1]
    w = img.shape[2]
    cut = 0.86
    img = img[:,int(h*(1-cut)):,:int(w*cut)]
    transform = v2.Compose([
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        v2.ColorJitter(saturation=(0.4, 2)),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize((0.5,), (0.5,))
    ])
    img = transform(img)
    return img

def bottom_right_crop(img):
    h = img.shape[1]
    w = img.shape[2]
    cut = 0.86
    img = img[:,int(h*(1-cut)):,int(w*(1-cut)):]
    transform = v2.Compose([
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        v2.ColorJitter(contrast=(0.5, 1.5)),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        # v2.Normalize((0.5,), (0.5,))
    ])
    img = transform(img)
    return img

# Define the data augmentation function
def data_augment(train_dataset, val_dataset):
    # Augmented datasets
    augmentations = [resize, central_crop, top_left_crop, top_right_crop, bottom_left_crop, bottom_right_crop]
    augmented_datasets = [CustomDataset(train_dataset, transform=v2.Lambda(aug)) for aug in augmentations]
    # Combine original and augmented datasets
    train_dataset = ConcatDataset(augmented_datasets)
    val_dataset = CustomDataset(val_dataset, transform=resize)
    return train_dataset, val_dataset



