import matplotlib.pyplot as plt
import numpy as np

NUM_CLASSES = 3


def imshow(img):
    """
    Plot an image.
    """
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    # PyTorch: the images have size [C, H, W].
    # NumPy: the images have size[H, W, C]
    # so here we change from size of pytorch to size of numpy.
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def classes_dist(dataset, dataset_name, is_loader):
    """
    Show class distribution in the given dataset.
    :param dataset: dataset of images and their labels.
    :param dataset_name: dataset name (for example "training set").
    :param is_loader: inicates if the dataset is loader type.
    :return:
    """
    counter_classes = [0] * NUM_CLASSES

    if is_loader:
        for _, batch_labels in dataset:
            for i in range(len(batch_labels)):
                counter_classes[batch_labels[i].item()] += 1
    else:
        for _, label in dataset:
            counter_classes[label] += 1

    font = {'weight': 'bold','size': 18}
    plt.rc('font', **font)
    plt.pie(x=counter_classes, autopct="%.1f%%", explode=[0.02] * 3, labels=["Car", "Truck", "Cat"], pctdistance=0.5)
    plt.title("Examples percentage per class\n in "+dataset_name, fontsize=19)
    plt.text(-2, -1.5, "Total: " + str(sum(counter_classes)) + " examples")
    plt.show()