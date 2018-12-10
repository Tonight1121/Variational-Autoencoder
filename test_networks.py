# Test the trained networks
# Created by yanrpi @ 2018-08-01
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
import filecmp
from os import path
from PIL import Image
import myresnet
import seaborn as sns
import autoencoder
import filecmp
import copy

######################################

# datafolder = '181007_50N_A'
# datafolder = '181031_multifc2'
# datafolder = '181031_multifc'
# datafolder = '181102_1channel'
# datafolder = '181107_avgtest'
# datafolder = 'cross_val_10_new'
# datafolder = 'cross_val_10'
# datafolder = '181114_slice'
datafolder = '181127_interest'
fn_roi_list = './data/roi_centers.txt'
center_list = np.loadtxt(fn_roi_list, dtype=np.int)
roi_w = 160 // 2
roi_h = 160 // 2
roi_d = 3 // 2
lower_b = -200
upper_b = 500

vec_mean = [0.17598677, 0.17181147, 0.16776074]
vec_std = [0.16603275, 0.16588687, 0.16528858]

# Just normalization for validation
data_transform = transforms.Compose([
        transforms.Resize(224),
        # transforms.Resize(int(224*1.5)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(vec_mean, vec_std)
    ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# network = ['ResNet-18', 'ResNet-34', 'ResNet-50', 'ResNet-101', 'ResNet-152']
network = ['ResNet-34']
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

# input an image array
# normalize values to 0-255
def array_normalize(input_array):
    max_value = np.max(input_array)
    min_value = np.min(input_array)
    # print('max {}, min {}'.format(max_value, min_value))
    k = 255 / (max_value - min_value)
    min_array = np.ones_like(input_array) * min_value
    normalized = k * (input_array - min_array)
    return normalized

# input the weights of neurons of fc layers
# output the normalized weights
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

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

def save_slices(inputs, list, preds):
    x_numpy = np.asarray(inputs)
    preds_numpy = np.asarray(preds)
    for i in range(0, x_numpy.shape[0]):
        slice = x_numpy[i, 0, :, :]
        normalized_slice = array_normalize(slice)
        cv2.imwrite('heatmaps/{}(L{}P{})_slice.jpg'.format(list[4][i],
                                                           list[3][i],
                                                           preds_numpy[i]), normalized_slice)

def save_heatmap(featuremap, img_id, class_id):
    ax = sns.heatmap(featuremap, vmin=0, vmax=255)
    fig = ax.get_figure()
    fig.savefig('heatmaps/{}_hpcls{}.jpg'.format(img_id, class_id))
    fig.clf()

def test_model(model, criterion):
    '''Test the trained models'''

    since = time.time()

    test_scores = []
    test_labels = []
    running_corrects = 0

    # Iterate over data.
    for inputs, input_patch, labels, agatston_scores, manual_scores, img_id in dataloader:
        print('imgid {}'.format(img_id))
        manual_scores = manual_scores.type(torch.FloatTensor)
        agatston_scores = agatston_scores.type(torch.FloatTensor)
        inputs = inputs.to(device)
        input_patch = input_patch.to(device)
        labels = labels.to(device)
        manual_scores = manual_scores.to(device)
        agatston_scores = agatston_scores.to(device)
        list = [inputs, manual_scores, agatston_scores, labels, img_id]
        # print(labels.shape)

        myresnet.myinputs = list
        # forward

        outputs, mu = model(inputs)
        inputs = inputs.type(torch.cuda.FloatTensor)
        loss = criterion(outputs, inputs)
        print('loss: {}'.format(loss))

        inputs_array = inputs.data.cpu().numpy()
        outputs_array = outputs.data.cpu().numpy()
        for i in range(0, inputs_array.shape[0]):
            save_sample(inputs_array[i, :, :, :], 'results/{}'.format(img_id[i]))
            save_sample(outputs_array[i, :, :, :], 'results/{}_outputs'.format(img_id[i]))

        # print('scores {}'.format(scores))
        # print('preds {}'.format(preds))
        # save_slices(inputs, list, preds)
        # print('slices have been saved!')
        # time.sleep(30)


    #time.sleep(10)

    loss = 0
    time_elapsed = time.time() - since
    #print('*'*10 + 'Test complete in {:.0f}m {:.0f}s'.format(
    #    time_elapsed // 60, time_elapsed % 60))
    #print()
    # print(test_scores)
    # print('test scores size {}'.format(len(test_scores)))
    # print(test_labels)
    # print('test labels size {}'.format(len(test_labels)))
    # print(acc)
    return test_scores, test_labels, loss


if __name__ == '__main__':
    k_tot = 10
    print(os.getcwd())
    avg_acc = 0
    #time.sleep(10)

    for net in network:
        if net == 'ResNet-18':
            base_model = myresnet.resnet18
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
            #continue
        else:
            print('The network of {} is not supported!'.format(net))

        #file.write('safdsaf')
        # for k in range(k_tot):
        for k in range(0, 1):
        # for k in range(1, 2):
            print('Testing fold {}/{} of {}'.format(k+1, k_tot, net))
            #file.write('Testing fold {}/{} of {}'.format(k+1, k_tot, net))
            #print(os.getcwd())
            data_dir = path.expanduser('~/tmp/{}/fold_{}'.format(datafolder, k))
            #image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
            #                                         data_transform)
            image_dataset = MortalityRiskDataset(os.path.join(data_dir, 'val'),
                                                      data_transform)

            dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=32,
                                                         shuffle=False, num_workers=0)

            dataset_size = len(image_dataset)

            class_names = image_dataset.classes

            #entry = [net]
            #print(entry)
            #print('what is entry')
            ft_acu = 0
            conv_acu = 0
            scratch_acu = 0

            #for mode in ['ft', 'conv', 'scratch']:
            for mode in ['scratch']:
                model_x = base_model(pretrained=False)
                # model_x = autoencoder.Autoencoder()
                # num_ftrs = model_x.fc.in_features

                # model_x.fc = nn.Linear(num_ftrs+1, 2)
                # model_x.fc = nn.Linear(4, 2)

                #data_dir = path.expanduser('~/tmp/my_cross_val_10/fold_0')
                fn_best_model = os.path.join(data_dir, 'best_vae_{}.pth'.format(net))
                model_x.load_state_dict(torch.load(fn_best_model))
                model_x.eval()

                model_x = model_x.to(device)
                criterion = nn.MSELoss()
                print(mode+': ', end='')
                scores, labels, myacc = test_model(model_x, criterion)
                avg_acc = avg_acc + myacc
                #entry.append(str(myacc))


                results = np.asarray([scores, labels])
                fn_results = os.path.join(data_dir, 'test_results_{}_{}.npy'.format(mode, net))
                np.save(fn_results, results)
                # print(scores.shape)
                # print(labels.shape)

    print('avg_acc = {:.4f}'.format(avg_acc/10))

"""
        ######################################################################
        # Finetuning the convnet
        # ----------------------
        #
        # Load a pretrained model and reset final fully connected layer.
        #

        #model_ft = models.resnet18(pretrained=True)
        model_ft = base_model(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 2)

        fn_best_model = os.path.join(data_dir, 'best_finetune_{}.pth'.format(net))
        model_ft.load_state_dict(torch.load(fn_best_model))
        model_ft.eval()

        model_ft = model_ft.to(device)

        scores, labels = test_model(model_ft)
        results = np.asarray([scores, labels])
        fn_results = os.path.join(data_dir, 'test_results_ft_{}.npy'.format(net))
        np.save(fn_results, results)


        ######################################################################
        # Use the convnet for feature extraction
        # ----------------------

        model_conv = base_model(pretrained=False)
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, 2)

        fn_best_model = os.path.join(data_dir, 'best_conv_{}.pth'.format(net))
        model_conv.load_state_dict(torch.load(fn_best_model))
        model_conv.eval()

        model_conv = model_conv.to(device)

        scores, labels = test_model(model_conv)
        results = np.asarray([scores, labels])
        fn_results = os.path.join(data_dir, 'test_results_conv_{}.npy'.format(net))
        np.save(fn_results, results)
        ######################################################################
"""