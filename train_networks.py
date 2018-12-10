# -*- coding: utf-8 -*-
"""
Adapted from Transfer Learning tutorial
==========================
**Original Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_

These two major transfer learning scenarios look as follows:

-  **Finetuning the convnet**: Instead of random initializaion, we
   initialize the network with a pretrained network, like the one that is
   trained on imagenet 1000 dataset. Rest of the training looks as
   usual.
-  **ConvNet as fixed feature extractor**: Here, we will freeze the weights
   for all of the network except that of the final fully connected
   layer. This last fully connected layer is replaced with a new one
   with random weights and only this layer is trained.

"""
# **Author**: yanrpi, created on 2018-07-29
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from os import path
import myresnet
import autoencoder
import tensorflow as tf
#from skimage import io
from PIL import Image
import copy

# datafolder = '181107_avgtest'
# datafolder = '181114_slice'
# datafolder = '181007_50N_A'
datafolder = '181127_interest'
fn_roi_list = './data/roi_centers.txt'
center_list = np.loadtxt(fn_roi_list, dtype=np.int)
roi_w = 160 // 2
roi_h = 160 // 2
roi_d = 3 // 2
lower_b = -200
upper_b = 500

epoch_scratch = 200
training_progress = np.zeros((epoch_scratch, 2))

plt.ion()   # interactive mode
# network = ['ResNet-18', 'ResNet-50', 'ResNet-101', 'ResNet-152']
network = ['ResNet-34']
# network = ['ResNet-18', 'ResNet-34']
# network = ['ResNet-50', 'ResNet-101', 'ResNet-152']
######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#

# for size 80x80
#vec_mean = [0.25374637, 0.24899475, 0.24366334]
#vec_std = [0.14036576, 0.14369524, 0.14501833]
# for size 160x160
vec_mean = [0.176, 0.172, 0.168]
vec_std = [0.166, 0.166, 0.165]

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224, scale=(0.6,0.8), ratio=(1.0,1.0)),
        transforms.Resize(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(vec_mean, vec_std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        # transforms.Resize(320),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(vec_mean, vec_std)
    ]),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


######################################################################
# Visualize a few images
# ^^^^^^^^^^^^^^^^^^^^^^
# Let's visualize a few training images so as to understand the data
# augmentations.

def save_sample(img1, img_name):
    slice1 = img1[0, :, :]
    slice2 = img1[1, :, :]
    slice3 = img1[2, :, :]
    slice1 = np.reshape(slice1, (slice1.shape[0], slice1.shape[1], 1))
    slice2 = np.reshape(slice2, (slice2.shape[0], slice2.shape[1], 1))
    slice3 = np.reshape(slice3, (slice3.shape[0], slice3.shape[1], 1))
    slice = np.concatenate((slice1, slice2), axis=2)
    slice = np.concatenate((slice, slice3), axis=2)
    slice = array_normalize(slice)
    cv2.imwrite('{}.jpg'.format(img_name), slice)

def array_normalize(input_array):
    max_value = np.max(input_array)
    min_value = np.min(input_array)
    # print('max {}, min {}'.format(max_value, min_value))
    k = 255 / (max_value - min_value)
    min_array = np.ones_like(input_array) * min_value
    normalized = k * (input_array - min_array)
    return normalized

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(vec_mean)
    std = np.array(vec_std)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
#inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
#out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

class MortalityRiskDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        """
        classes, class_to_idx = find_classes(root_dir)
        samples = make_dataset(root_dir, class_to_idx, IMG_EXTENSIONS)

        self.root_dir = root_dir
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        img_path, target = self.samples[idx]

        # find last .png
        # extract the code right before it
        img_id = img_path[-19: -13]
        agatston_score = float(img_path[-9]) * 2 / 3 - 1
        manual_score = float(img_path[-7]) * 2 / 3 - 1
        # life_label = float(img_path[-9])
        idx = np.where(center_list[:, 0] == int(img_id))[0][0]
        center = center_list[idx, 1:]

        # print('img_path {}'.format(img_path))
        # print('center list {}'.format(center_list))

        #img_name = os.path.join(self.root_dir,
        #                        self.image_filenames[idx])
        #image = io.imread(img_path)
        with open(img_path, 'rb') as f:
            # print('what is f {}'.format(f))
            # time.sleep(30)
            img = Image.open(f)
            image = img.convert('RGB')

            xl = center[0] - roi_w - 1
            xu = center[0] + roi_w
            yl = center[1] - roi_h - 1
            yu = center[1] + roi_h
            patch = image.crop((xl, yl, xu, yu))


        # time.sleep(30)
        # plt.imshow(patch)
        # time.sleep(30)


        if self.transform:
            image = self.transform(image)
            patch = self.transform(patch)

        return image, patch, target, agatston_score, manual_score, img_id

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

def train_model(model, criterion, optimizer, scheduler, fn_save, num_epochs=25):
    since = time.time()

    #best_model_wts = copy.deepcopy(model.state_dict())
    least_loss = 10000.0
    best_ep = 0

    tv_hist = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, input_patch, labels, agatston_scores, manual_scores, img_id in dataloaders[phase]:
                # Get images from inputs
                #print('*'*10 + ' printing inputs and labels ' + '*'*10)

                manual_scores = manual_scores.type(torch.FloatTensor)
                agatston_scores = agatston_scores.type(torch.FloatTensor)

                inputs = inputs.to(device)
                input_patch = input_patch.to(device)
                labels = labels.to(device)
                manual_scores = manual_scores.to(device)
                agatston_scores = agatston_scores.to(device)
                list = [inputs, manual_scores, agatston_scores, labels]

                # print(img_id)
                # print(manual_scores)
                # print(agatston_scores)
                # print(labels)
                # time.sleep(10)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    myresnet.myinputs = list
                    # print('{}'.format(list.__sizeof__()))
                    # outputs, mu, layvar = model(inputs)
                    outputs, code = model(inputs)
                    inputs = inputs.type(torch.cuda.FloatTensor)

                    inputs_array = inputs.data.cpu().numpy()
                    outputs_array = outputs.data.cpu().numpy()
                    save_sample(inputs_array[0, :, :, :], 'inputs')
                    save_sample(outputs_array[0, :, :, :], 'outputs')

                    # inputs = inputs.type(torch.cuda.LongTensor)
                    # outputs = outputs.type(torch.cuda.LongTensor)
                    # print('input size {}'.format(inputs.shape))
                    # print('input type {}'.format(inputs.dtype))
                    # print('output size {}'.format(outputs.shape))
                    # print('output type {}'.format(outputs.dtype))
                    loss = criterion(outputs, inputs)
                    # print('loss shape {}'.format(loss.shape))
                    # print('loss is {}'.format(loss))
                    # time.sleep(1)
                    # time.sleep(30)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]

            tv_hist[phase].append([epoch_loss])

            # deep copy the model
            if phase == 'val' and epoch_loss <= least_loss:
                least_loss = epoch_loss
                best_ep = epoch
                #best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), fn_save)

        print('ep {}/{} - Train loss: {:.4f}, Val loss: {:.4f}'.format(
            epoch + 1, num_epochs, 
            tv_hist['train'][-1][0],
            tv_hist['val'][-1][0]))
        training_progress[epoch][0] = tv_hist['train'][-1][0]
        # training_progress[epoch][1] = tv_hist['train'][-1][1]
        training_progress[epoch][1] = tv_hist['val'][-1][0]
        # training_progress[epoch][3] = tv_hist['val'][-1][1]
        #print('-' * 10)

        #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        #    phase, epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print('*'*10 + 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('*'*10 + 'Least Val Loss: {:4f} at epoch {}'.format(least_loss, best_ep))
    print()

    # load best model weights
    #model.load_state_dict(best_model_wts)
    #return model
    return tv_hist


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


#
# %% 10-fold cross validation
#

k_tot = 10

for net in network:
    epoch_ft = 100
    epoch_conv = 100
    if net == 'ResNet-18':
        base_model = myresnet.resnet18
        #continue
    elif net == 'ResNet-34':
        base_model = myresnet.resnet34
        #continue
    elif net == 'ResNet-50':
        base_model = myresnet.resnet50
        #continue
    elif net == 'ResNet-101':
        base_model = myresnet.resnet101
        #continue
    elif net == 'ResNet-152':
        base_model = myresnet.resnet152
    else:
        print('The network of {} is not supported!'.format(net))

    # for k in range(2, k_tot):
    for k in range(0, 1):
        print('Cross validating fold {}/{} of {}'.format(k+1, k_tot, net))
        data_dir = path.expanduser('~/tmp/{}/fold_{}'.format(datafolder, k))
        #image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
        #                                        data_transforms[x])
        #                for x in ['train', 'val']}
        image_datasets = {x: MortalityRiskDataset(os.path.join(data_dir, x),
                                                data_transforms[x])
                        for x in ['train', 'val']}

        #print(image_datasets)
        #time.sleep(10)
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                    shuffle=True, num_workers=0)
                    for x in ['train', 'val']}

        print('size of dataloader: {}'.format(dataloaders.__sizeof__()))
        #time.sleep(10)
        #print(dataloaders.__len__())
        #time.sleep(10)
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        #print(dataset_sizes)
        class_names = image_datasets['train'].classes
        #print(class_names)


        ######################################################################
        # Train from scratch

        #model_ft = models.resnet18(pretrained=True)
        model_ft = base_model(pretrained=False)
        # model_ft = autoencoder.Autoencoder()
        # model_ft = myresnet.VAE()
        # model_ft = myresnet.CNN_VAE()
        # model_ft = myresnet.ResNet_VAE()
        model_ft.cuda()
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, 2)
        # model_ft.fc = nn.Linear(num_ftrs, 2)

        model_ft = model_ft.to(device)

        # criterion = nn.CrossEntropyLoss()
        criterion = nn.MSELoss()
        # criterion = nn.L1Loss()
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)
        # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.00001)


        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.9)

        # Train and evaluate
        fn_best_model = os.path.join(data_dir, 'best_vae_{}.pth'.format(net))
        hist_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                            fn_best_model, num_epochs=epoch_scratch)
        fn_hist = os.path.join(data_dir, 'hist_vae_{}.npy'.format(net))
        np.save(fn_hist, hist_ft)
        txt_path = path.join(data_dir, 'vae_training_progress.txt')
        np.savetxt(txt_path, training_progress)
        ######################################################################
