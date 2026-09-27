import numpy as np
import torch
import math
from sklearn import metrics
from pylab import *

##################    Class weights    ###################################################


def calc_cls_weights_log(dataset, config):
    """
    Function to calculate class weights depending on:
    https://medium.com/@meet_patel/notes-on-implementation-of-cross-entropy-loss-2a8e3408413c
    depending on all the labels that are available in the dataset
    :param dataset: pytorch dataset, based on class 'Eval_Dataset'
    :param config: All parameters that are used during training (from args.py)
    :return: cls_weights: np.array of size (number_of_classes) with the weights for each of the classes
                            example: [3.2, 1.5, 1.6, 3.9, 1.3]
    """
    anz_Pixel = 0
    cls_weights = np.zeros(13, dtype=np.float32)
    cls_percent = np.zeros(13, dtype=np.float32)

    for i in range(dataset.__len__()):
        current_ = dataset.__getitem__(i)
        current = current_["labels"]

        current_np = current.numpy()
        # Totol number of pixels
        anz_Pixel = dataset.__len__() * current_np.shape[0] * current_np.shape[0]

        unique, counts = np.unique(current_np, return_counts=True)  # [0 1 2 3 4 5 6] []

        for j in range(len(unique)):
            clss = unique[j]
            cls_weights[clss] += counts[j]

    # print('occurences: ', cls_weights)
    # [ 105320.       0.  280732.       0.   40152.       0.  497384.  127176. 0.  200652. 1350892.       0.   19132.]
    # cls_percent = cls_weights / anz_Pixel
    # [0.04017639 0.         0.10709076 0.         0.01531677 0. 0.18973694 0.04851379 0.         0.07654266 0.5153244  0. 0.00729828]
    # print('percent:', cls_percent)

    for i in range(len(cls_weights)):
        if cls_weights[i] != 0:
            cls_weights[i] = math.log(anz_Pixel / cls_weights[i])
        else:
            cls_weights[i] = 0
    max_weight = np.max(cls_weights)
    cls_weights = cls_weights / max_weight
    cls_weights[0] = 0.1
    print("Computed new class weights: ", cls_weights)
    return cls_weights


#################     Accuracy metrices     #############################################


def calc_IoU(CM, add_zero_bef=False):
    """
    Function that calculates the IoU from a given confusion matrix.
    :param CM: np array in which the confusion matrix was safed before
    :param add_zero_bef: bool, if True a IoU is added for class 0

    :return: IoUs: IoUs of all classes
    """
    if np.sum(CM) == 0:
        return 0
    IoUs = []
    if add_zero_bef:
        IoUs.append(0.0)
    for i in range(len(CM)):
        support_current = np.sum(CM[i, :])
        TP = CM[i, i]
        if TP == 0:
            IoU = 0
        else:
            FP = np.sum(CM[:, i]) - TP
            FN = support_current - TP
            # prec = TP / float(TP + FP)
            # recall = TP / float(TP + FN)
            IoU = TP / float(TP + FP + FN)
        IoUs.append(IoU)
    return IoUs


def calc_mean_F1(CM):
    """
    Function that calculates the mean F1-score from a given confusion matrix.
    :param CM: np array in which the confusion matrix was safed before
    :param outclass: bool, if 1 the first row and column is deleted before the mean F1 score is calculated
                    this is helpfull if the 0th class is a background class that shall not be evaluated
    :return: np.mean(fscores): mean f1-score
    """
    if np.sum(CM) == 0:
        return 0
    fscores = []
    for i in range(len(CM)):
        support_current = np.sum(CM[i, :])
        TP = CM[i, i]
        FP = np.sum(CM[:, i]) - TP
        FN = support_current - TP
        prec = TP / float(TP + FP)
        recall = TP / float(TP + FN)
        f_score = 2 * prec * recall / (prec + recall)
        fscores.append(f_score)
    return np.mean(fscores)


def calc_prec_recall(CM):
    """
    Function that calculates precision and recall for all classes for a given cm
    """
    if np.sum(CM) == 0:
        return 0
    precisions = []
    recalls = []
    for i in range(len(CM)):
        support_current = np.sum(CM[i, :])
        TP = CM[i, i]
        FP = np.sum(CM[:, i]) - TP
        FN = support_current - TP
        prec = TP / float(TP + FP)
        recall = TP / float(TP + FN)
        precisions.append(prec)
        recalls.append(recall)
    return precisions, recalls


def calc_OA(CM):
    """
    Function that calculates the overall accuracy from a given confusion matrix
    :param CM: np array in which the confusion matrix was saved
    :return: returns the overall accuracy
    """
    if np.sum(CM) == 0:
        return 0
    return np.trace(CM) / np.sum(CM)  # OA


def calculate_conf_matrix_sclearn(preds, labels, classes, ignore_firstclass=False):
    """
    Function to calculate a confusion matrix based on a given prediction, compared to a reference (labels)
    :param preds: numpy array of predictions. size: n x height x width
    :param labels: numpy array of true classes. size: n x height x width
    :param classes: array like shape (n_classes). List of labels to index in the matrix
    :return: con_mat: numpy array of size n_classes x n_classes with total number of pixels in the entries
    """
    conf_mat = metrics.confusion_matrix(
        labels.flatten(), preds.flatten(), labels=classes
    )
    if ignore_firstclass == 1:
        conf_mat = conf_mat[1:, 1:]
    return conf_mat


def calculate_conf_matrix(preds, labels, classes, ignore_firstclass=False):
    """
    Function to calculate a confusion matrix based on a given prediction, compared to a reference (labels)
    :param preds: numpy array of predictions. size: n x height x width
    :param labels: numpy array of true classes. size: n x height x width
    :param classes: array like shape (n_classes). List of labels to index in the matrix
    :return: con_mat: numpy array of size n_classes x n_classes with total number of pixels in the entries
    """
    if preds is None:
        print("preds are none")
    if labels is None:
        print("labels are none")
    if classes is None:
        print("classes is none")
    # conf_mat = metrics.confusion_matrix(labels.flatten(), preds.flatten(), labels=classes)
    conf_mat = comp_confmat(labels, preds, classes)
    if ignore_firstclass == 1:
        conf_mat = conf_mat[1:, 1:]
    return conf_mat


def comp_confmat(actual, predicted, classes):
    # extract the different classes
    # classes = np.unique(actual)

    # initialize the confusion matrix
    confmat = np.zeros((len(classes), len(classes)))

    # loop across the different combinations of actual / predicted classes
    for i in range(len(classes)):  # schleife über label
        for j in range(len(classes)):  # schleife über pred.
            # count the number of instances in each combination of actual / predicted classes
            confmat[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))

    return confmat


############### Loss functions #########################################################
class Cross_Entropy_Loss_timeseries:
    """
    compute cross entropy loss for output time series
    input=
    output=
    """

    def __init__(self, class_weights, ignore_index=False):
        self.class_weight = class_weights

    def __call__(self, *args, **kwargs):
        return


class FocalCrossEntropyLoss:
    """
    compute focal loss according to the prob of the sample.
    loss= -(1-p)^gamma*log(p)
    self.criterion = functions.FocalCrossEntropyLoss(self.ncls, gamma=)
    gamma>0 reduces the rel. loss function for well-classified samples (p > 0.5), putting more focus on hard, misclassified samples
    """

    def __init__(self, n_cls, weights, gamma=0.0):
        """
        Initialization of an object of this class
        :param n_cls: number of classes
        :param weights: numpy array with the weights that shall be used for the individual classes
        :param gamma: float, parameter gamme for focal loss
        """
        self.n_cls = n_cls
        self.gamma = gamma
        self.weights = weights

    def __call__(self, logits, labels):
        labels_flat = labels.view(-1, 1)  # ----------------- [NHW ,1]
        logits_t = logits.transpose(1, 2).transpose(2, 3)
        logits_flat = logits_t.reshape(-1, self.n_cls)  # -------- [NHW , ncls]

        labels_oh = torch.zeros_like(logits_flat)  # construct one-hot encoded labels
        labels_oh.scatter_(1, labels_flat, 1.0)

        eps = 1e-12
        softmax = torch.softmax(logits_flat, 1)  # N x C

        inv_softmax = torch.ones_like(softmax) - softmax
        inv_softmax = inv_softmax.detach_()
        if self.gamma:
            inv_softmax = torch.pow(inv_softmax, self.gamma)
        if self.weights[0] == 0:
            loss = -torch.mean(labels_oh * torch.log(softmax + eps) * inv_softmax)
        else:
            loss = -torch.mean(
                self.weights * labels_oh * torch.log(softmax + eps) * inv_softmax
            )
        return loss
