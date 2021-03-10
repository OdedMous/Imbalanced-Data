import copy
import models
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import visualizations

def adversarial_optimizing_noise(model, org_img, true_label, target_label, regularization="l1"):
    """
    Creates an adversarial image by optimizing some noise, and add it to some original image.
    :param model: the trained model we want to fool.
    :param org_img: original image. to it we want to add the noise in order to create the adversarial image.
    :param true_label: the gold label of org_image.
    :param target_label: the label we want the trained model will mistakly classify it for the adversarial image.
    :param regularization: which norm to use in order to keep the noise as low as possibale.
    :return: noise - the noise we should add to original image in order to create an adversarial image
             pred_adversarial_label - the last label the trained model predicted to the noise image
                                      in the noise optimization iterations.
    """

    # necessary pre-processing
    target_label = torch.LongTensor([target_label]) #
    org_img = org_img.unsqueeze(0)     # add batch diminsion to org_image

    # Init value of noise and make its gradients updatable
    noise = nn.Parameter(data=torch.zeros(1, 3*32*32), requires_grad=True) # gray image
    #noise = nn.Parameter(data=torch.ones(1, 3*32*32), requires_grad=True) # white image
    #noise = nn.Parameter(data=torch.randn(1, 3*32*32), requires_grad=True)  # gaussion noise

    # Check classification before modification
    pred_label = np.argmax(model(org_img).data.numpy())
    if true_label != pred_label:
        print("WARNING: IMAGE WAS NOT CLASSIFIED CORRECTLY")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=[noise], lr=0.001, momentum=0.9)

    # Noise optimization
    iterations = 30000
    for iteration in range(iterations):

        optimizer.zero_grad()
        output = model(org_img + noise.view((1,3,32,32)))
        loss = criterion(output, target_label)

        if regularization == "l1":
            adv_loss = loss + torch.mean(torch.abs(noise))
        elif regularization == "l2":
            adv_loss = loss + torch.mean(torch.pow(noise, 2))
        else:
            adv_loss = loss

        adv_loss.backward()
        optimizer.step()

        # keep optimizing until we get that the predicted label is the target label
        pred_adversarial_label = np.argmax(model(org_img).data.numpy())
        if pred_adversarial_label == target_label:
            break

    if iteration == iterations-1:
        print("Warning: optimization loop ran for the maximum iterations. The result may not be correct")

    return noise.view((3,32,32)).detach(), pred_adversarial_label

def FGSM(model, org_img, true_label):
    """
    Creates an adversarial image by Fast Gradient Sign Method.
    :param model: the trained model.
    :param org_img: original image. to it we want to add the noise in order to create the adversarial image.
    :param true_label: the gold label of org_image.
    :return: adversarial_img,
             noise - the noise used to create the adversarial image
            y_pred_adversarial
    """

    true_label = Variable(torch.LongTensor(np.array([true_label])), requires_grad=False)

    org_img = org_img.unsqueeze(0) # add batch diminsion
    org_img = Variable(org_img, requires_grad=True)  # set org_img as parameter (cuz we need its gradient)

    # Classification before Adv
    pred_label = np.argmax(model(org_img).data.numpy())

    criterion = nn.CrossEntropyLoss()

    # Forward pass
    output = model(org_img)
    loss = criterion(output, true_label)
    loss.backward()  # obtain gradients on org_img

    # Add perturbation
    epsilon = 0.01 #0.01 # 0.15
    x_grad = torch.sign(org_img.grad.data)
    noise = epsilon * x_grad
    adversarial_img = torch.clamp(org_img.data + noise, 0, 1)

    # Classification after optimization
    y_pred_adversarial = np.argmax(model(Variable(adversarial_img)).data.numpy())

    return adversarial_img.squeeze(0), noise.squeeze(0), y_pred_adversarial


def create_adversarial_img(path, org_img, true_label):
    """
    Creates an adversarial image, and display it. We do it with 2 different methods.
    :param path: a path for the trained model.
    :param org_img: original image. to it we want to add the noise in order to create the adversarial image.
    :param true_label: the gold label of org_image.
    """

    # Load trained model
    trained_net = models.SimpleModel()
    trained_net.load(path=path)
    trained_net.eval()

    # show original image
    visualizations.imshow(org_img)

    # Adversarial method 1

    # Copy the model so the original trained network wont change while we creating
    # the adversarial image
    model_copy = copy.deepcopy(trained_net)
    model_copy.eval()

    noise, adv_label = adversarial_optimizing_noise(model_copy, org_img, true_label=0, target_label=2, regularization="l1")

    visualizations.imshow(noise) # show noise
    visualizations.imshow(org_img+noise) # show adversarial image
    out = trained_net((org_img+noise).unsqueeze(0))
    print("true label:", true_label, "adv_label:", adv_label, "trained_net label:", out)


    # Adversarial method 2

    model_copy2 = copy.deepcopy(trained_net)
    adver_img, noise2, adv_label_2 = FGSM(model_copy2, org_img, true_label=0)

    visualizations.imshow(noise2) # show noise
    visualizations.imshow(adver_img) # show adversarial image
    out = trained_net(adver_img.unsqueeze(0))
    print("true label:", true_label, "adv_label:", adv_label_2, "trained_ned label:", out)

