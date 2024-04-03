import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
import seaborn as sns
import pandas as pd


def plot_confusion_matrix(cm, savefile):
    fig, ax = plt.subplots()
    cm = np.around(cm, decimals=6)
    print(cm)
    if len(cm) < 2:
        cm = np.array([[0, 0], [0, cm[0][0]]])
    df_cm = pd.DataFrame(cm, index=[i for i in ['Normal', 'Anomaly']],
                         columns=[i for i in ['Normal', 'Anomaly']])
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True, fmt="g", annot_kws={"size": 16})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(savefile, bbox_inches="tight")
    plt.close('all')


def plot_accuracy(history, savefile):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['accuracy', 'val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy in percent')
    plt.savefig(savefile, dpi=300, bbox_inches="tight")
    plt.close('all')


def plot_loss(history, savefile):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(savefile, dpi=300, bbox_inches="tight")
    plt.close('all')


def plot_roc(tpr, fpr, roc_auc, savefile):
    fig, ax = plt.subplots()
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, lw=1, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='lime', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(savefile, dpi=300, bbox_inches="tight")
    plt.close('all')


def plot_losses(gen_loss, dis_loss, savefile):
    fig, ax = plt.subplots()
    plt.plot(dis_loss, label='Discriminator')
    plt.plot(gen_loss, label='Generator')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(savefile, dpi=300, bbox_inches="tight")
    plt.close('all')


def plot_precision_recall(y_test, y_preds, savefile):
    fig, ax = plt.subplots()
    print(y_test.shape)
    print(y_preds.shape)
    skplt.metrics.plot_precision_recall(y_test, y_preds)
    plt.savefig(savefile, dpi=300, bbox_inches="tight")
    plt.close('all')
