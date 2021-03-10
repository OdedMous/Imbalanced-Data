import sys
import numpy as np
from collections import Counter
import torch
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

import dataset
import models
import visualizations
import adversarial

BATCH_SIZE = 4
NUM_CLASSES = 3

def evaluate_model(model, test_loader):
    """
    Evaluates the given mode by displaying Accuracy, Recall and Precision per class.
    :param model: a trained model.
    :param test_loader: loader for test set.
    """
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("Accuracy of the network on the " + str(total) + " test images: " + str(100*correct / total) + "%")

    classes = ("car", "truck", "cat")
    class_correct = list(0. for i in range(NUM_CLASSES))
    class_total = list(0. for i in range(NUM_CLASSES))
    list_labels = []
    list_preds = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels)
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

                list_labels.append(labels[i].item())
                list_preds.append(predicted[i].item())

    for i in range(NUM_CLASSES):
        print('Accuracy of %5s : %5d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    # Display confusion matrix
    conf_mat = confusion_matrix(y_pred=list_preds, y_true=list_labels)
    df_cm = pd.DataFrame(conf_mat, index=classes,columns=classes)
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 15}, fmt="d", linewidths=.5)  # font size
    plt.xlabel('Predicted label', fontsize=15)  # x-axis label with fontsize 15
    plt.ylabel('True label', fontsize=15)  # y-axis label with fontsize 15
    plt.show()

    # Display metrics (Recall, Precision and F1) per class + Total Accuracy.
    print(metrics.classification_report(y_pred=list_preds, y_true=list_labels, digits=3, target_names=["Car", "Truck", "Cat"]))


def train_model(model, train_loader, examples_num, epochs_num=10):
    """
    Trains the given model.
    :param model: a model.
    :param train_loader: train set loader.
    :param examples_num: number of examples in train_loader.
    :param epochs_num: number of iteration the training will pass over all the dataset.
    """
    weights = torch.tensor([examples_num/(88*examples_num/100), examples_num/(8*examples_num/100), examples_num/(4*examples_num/100)]).float()
    criterion = nn.CrossEntropyLoss() # TODO insert weight=weights
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    losses = []

    for epoch in range(epochs_num):  # loop over the dataset multiple times

        if epoch%4 == 0:
            print(epoch,"/",epochs_num)

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        losses.append(running_loss/examples_num)

    print('Finished Training')
    plt.plot(losses)
    plt.title('loss vs epochs')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()


def create_weights_for_balanced_classes(images, nclasses):
    """
    Returns a list of weights. Each weight is correspond to an example in the dataset, and for minority examples
    we give a higher weight. So  using these weights cause the classes in the dataset to be balanced.
    :param images: dataset.
    :param nclasses: number of classes in the dataset.
    :return: list of weights.
    """
    count = [0] * nclasses   # histogram by classes
    for item in images:
        count[item[1]] += 1

    # number of examples in dataset
    N = float(sum(count))

    # calculate weight for each class
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])

    # calculate weight for each example (image)
    weights = [0] * len(images)
    for idx, val in enumerate(images):
        weights[idx] = weight_per_class[val[1]]
    return weights


def augment_data(path, train_data):
    """
    Do data augmentation on the given data, and returns the original data + augmentations of this data.
    :param path: path to training set.
    :param train_data: training set.
    :return:
    """

    transforms_list = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

    transformed_train_data = dataset.get_dataset_as_torch_dataset(path=path, transform=transforms_list)
    return torch.utils.data.ConcatDataset([transformed_train_data, train_data])


def synthetic_data(train_data):
    """
    Creates suntetic data using SMOTE method.
    :param train_data:
    :return:
    """
    X = np.asarray([example[0].numpy() for example in train_data])
    y = [example[1] for example in train_data]

    w, h, c = X.shape[1], X.shape[2], X.shape[3]
    smote = SMOTE(random_state = 1000)
    X_resampled, y_resampled = smote.fit_sample(X.reshape(X.shape[0], w*h*c), y)
    X_resampled = X_resampled.reshape(X_resampled.shape[0], w, h, c)

    resampled = []
    for i in range(X_resampled.shape[0]):
        resampled.append((torch.from_numpy(X_resampled[i]),y_resampled[i]))
    return resampled


def main():

    # Load training set
    train_data_path = sys.argv[1]
    train_data = dataset.get_dataset_as_torch_dataset(path=train_data_path)

    #  dataset with augmentation
    increased_train_data = augment_data(train_data_path, train_data)

    # Weights for balance the training data
    weights = create_weights_for_balanced_classes(train_data, NUM_CLASSES)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))

    # Load test set
    test_data = dataset.get_dataset_as_torch_dataset(path=sys.argv[2])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    # Plot orignal datasets class distribution
    # visualizations.classes_dist(train_data, "training set", is_loader=False)
    # visualizations.classes_dist(test_data, "test set", is_loader=False)

    # ---------------**** Options for training set************-----------------
    #---------------- One should use only a single train_loader ----------------

    # original train set
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    visualizations.classes_dist(train_loader, "training set", is_loader=True)

    # train set with sampler
    #train_loader = torch.utils.data.DataLoader(train_data, sampler=sampler, batch_size=BATCH_SIZE)
    #visualizations.classes_dist(train_loader, "training set", is_loader=True)

    # train set with augmentation
    #train_loader = torch.utils.data.DataLoader(increased_train_data, batch_size=BATCH_SIZE, shuffle=True)
    #visualizations.classes_dist(train_loader, "training set", is_loader=True)

    # train set with augmentation + with sampler
    #train_loader = torch.utils.data.DataLoader(increased_train_data,sampler=sampler, batch_size=BATCH_SIZE)
    #visualizations.classes_dist(train_loader, "training set", is_loader=True)

    # train set with SMOTE
    #synthetic_train__data = synthetic_data(train_data) # orignal train set + synthetic data
    #train_loader = torch.utils.data.DataLoader(synthetic_train__data, batch_size=BATCH_SIZE, shuffle=True)
    #visualizations.classes_dist(train_loader, "training set", is_loader=True)

    # data which contains  syntatic train data + test data
    #final_data = torch.utils.data.ConcatDataset([synthetic_train__data, test_data])
    #train_loader = torch.utils.data.DataLoader(final_data, batch_size=BATCH_SIZE, shuffle=True)
    #visualizations.classes_dist(train_loader, "training set", is_loader=True)

    # ---------------************************************-----------------

    # Evaluate baseline model
    baseline_net = models.SimpleModel()
    baseline_net.load(sys.argv[4])
    evaluate_model(baseline_net, test_loader)

    # Train my model
    my_net = models.SimpleModel()
    train_model(my_net, train_loader, examples_num=len(train_data), epochs_num=20)

    # Evaluate my model
    evaluate_model(my_net, test_loader)

    # Save my model
    trained_model_path = sys.argv[3]
    # my_net.save(path=trained_model_path)

    # Create an adversarial example
    org_img, true_label = test_data[0][0], test_data[0][1]
    adversarial.create_adversarial_img(trained_model_path, org_img, true_label)




main()
