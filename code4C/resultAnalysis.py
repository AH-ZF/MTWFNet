"""
Mainly analyze the results of model training: loss curve, F1-score, Acc, etc.
"""

import matplotlib.pyplot as plt
import os, datetime, math
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torchstat import stat, ModelStat, report_format
from matplotlib import rcParams

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def confusionMatrix(gt, pred, show=False):
    TP = 0
    for i in range(len(gt)):
        if gt[i] == pred[i] and gt[i] == 1:
            TP += 1
    FP = sum(pred) - TP
    FN = sum(gt) - TP
    TN = len(gt) - TP - FP - FN

    if TP == 0:
        f1_score = 0
        pass
    else:
        f1_score = (2 * TP) / (2 * TP + FP + FN)
        pass
    num_samples = len([x for x in gt if x == 1])
    if num_samples != 0:
        recall = TP / num_samples
    else:
        recall = 0
        pass
    return f1_score, recall


def paper_metric(y_true, y_pred):
    lenghtC = len(set(y_true))
    TP = []
    FP = []
    FN = []
    TN = []
    Fi = []
    Ris = []
    PiS = []
    for i in range(lenghtC):
        sums = 0
        sumstn = 0
        for j in range(len(y_pred)):
            if i == y_true[j] and i == y_pred[j]:
                sums += 1
                pass
            if i != y_true[j] and i != y_pred[j]:
                sumstn += 1
                pass
            pass

        TP.append(sums)
        FP.append(y_pred.count(i) - sums)
        FN.append(y_true.count(i) - sums)
        TN.append(sumstn)
        if sums == 0:
            Pis = 0
            pass
        else:
            Pis = sums / (y_pred.count(i))
            pass

        Ri = sums / (y_true.count(i))
        Ris.append(Ri)
        PiS.append(Pis)
        if Pis == 0 and Ri == 0:
            Fis = 0
            pass
        else:
            Fis = 2 * Pis * Ri / (Pis + Ri)
            pass

        Fi.append(Fis)
        pass
    # allR = sum(TP) / (sum(TP) + sum(FN))
    # allP = sum(TP) / (sum(TP) + sum(FP))
    allR = sum(Ris) / len(Ris)
    allP = sum(PiS) / len(PiS)
    uf1 = sum(Fi) / len(Fi)
    uar = sum(Ris) / len(Ris)
    accs = sum(TP) / (sum(TP) + sum(FP))

    if allR == 0 and allP == 0:
        print('Error: An error occurred in calculating the F1 score for multiple classifications, with the denominator being 0!')
        pass
    else:
        f1_score = 2 * allR * allP / (allR + allP)
        pass
    return accs, f1_score, uf1, uar


def recognition_evaluation(final_gt, final_pred, show=False):
    label_dict = {'negative': 0, 'positive': 1, 'surprise': 2}

    # Display recognition result
    f1_list = []
    ar_list = []
    for emotion, emotion_index in label_dict.items():
        gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
        pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
        try:
            f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
            f1_list.append(f1_recog)
            ar_list.append(ar_recog)
        except Exception as e:
            pass
    UF1 = np.mean(f1_list)
    UAR = np.mean(ar_list)
    return UF1, UAR



def GENlosscurve(x, y, savepath, index):
    LossCueve = os.path.join(savepath,
                             "LossCueve(" + str(index) + f"){datetime.datetime.now():%Y-%m-%d-%H-%M-%S}" + ".png")
    plt.figure("Loss_curve")
    plt.title("LOSS Curve")
    plt.plot(x, y, "r-x", markersize=5, lw=1, label="epoch loss")
    plt.xlabel("epoch", fontsize=10)
    plt.ylabel("epoch_loss", fontsize=10)
    plt.legend()
    for i in x:
        plt.text(i, y[i], "({},{:.3f})".format(i, y[i]))
        # plt.annotate("({},{:.2f})".format(i, y[i - 1]), xy=(i, y[i - 1]))
    plt.savefig(LossCueve)
    plt.close()
    pass



def GENxycurve(x, y, savepath, index, imgnname):
    Cueve = os.path.join(savepath,
                         imgnname + "(" + str(index) + ").png")
    plt.figure("Curve")
    plt.title(imgnname)
    plt.plot(x, y, "r-x", markersize=5, lw=1, label="Epoch " + imgnname)

    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Epoch_" + imgnname, fontsize=10)
    plt.grid(c='k', ls='-', lw=0.4)
    plt.legend()
    for i in range(len(x) - 10, len(x), 2):
        # i += 2
        plt.text(i, y[i], "({},{:.3f})".format(i, y[i]))
        pass
    locy = 0.5
    for i in range(len(x) - 5, len(x)):
        # i += 2
        # plt.text(len(x) - 5, locy, "({},{:.3f})".format(i, y[i]), c='r', backgroundcolor='w')
        plt.text(len(x) - 5, locy, "({},{:.3f})".format(i, y[i]), c='r')
        locy += 0.02
        pass

    plt.savefig(Cueve)
    plt.close()


    pass






def GENconfusion_matrix(y_true, y_pred, savepath, index, imgnname):
    config = {
        "font.family": 'Times New Roman',
        "font.size": 9,
        "mathtext.fontset": 'stix',
        "font.weight": 'bold',
        "figure.dpi": 600
    }

    configxylabel = {
        "family": 'Times New Roman',
        "size": 9,
        "weight": 'bold'
    }
    rcParams.update(config)
    confMfig = os.path.join(savepath,
                            imgnname + "(" + str(index) + ")" + ".png")
    configpng = os.path.join(savepath,
                             imgnname + "(" + str(index) + ")" + 'Normalize' + ".png")
    configtif = os.path.join(savepath,
                             imgnname + "(" + str(index) + ")" + 'Normalize' + ".tif")

    confMfigpdf = os.path.join(savepath,
                               imgnname + "(" + str(index) + ")" + 'Normalize' + ".pdf")
    confMfigeps = os.path.join(savepath,
                               imgnname + "(" + str(index) + ")" + 'Normalize' + ".eps")

    classes = ["Negative", "Positive", "Surprise", 'Others']

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(include_values=True, cmap=plt.cm.Reds, colorbar=True)

    plt.savefig(confMfig, dpi=600)
    plt.close()
    np.set_printoptions(precision=4, suppress=True)
    cmN = np.zeros((cm.shape[0], cm.shape[1]))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cmN[i][j] = cm[i][j] / sum(cm[i])
            pass
        pass
    dispN = ConfusionMatrixDisplay(confusion_matrix=cmN, display_labels=classes)
    dispN.plot(include_values=True, cmap=plt.cm.Reds, colorbar=True, values_format='.4f')
    plt.xlabel('Predicted label', fontdict=configxylabel)
    plt.ylabel('True label', fontdict=configxylabel)
    plt.savefig(configpng, dpi=600)
    plt.savefig(configtif, dpi=600)
    plt.close()

    pass



def GENprecision_score(y_true, y_pred):
    precisionmicro = precision_score(y_true, y_pred, average='micro')
    return precisionmicro


def GENaccuracy_score(y_true, y_pred):
    Acc = accuracy_score(y_true, y_pred, normalize=True)
    return Acc



def GENf1_score(y_true, y_pred):
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    return micro_f1



def self_f1(y_true, y_pred):
    lenghtC = len(set(y_true))
    TP = []
    FP = []
    FN = []
    TN = []
    for i in range(lenghtC):
        sums = 0
        sumstn = 0
        for j in range(len(y_pred)):
            if i == y_true[j] and i == y_pred[j]:
                sums += 1
                pass
            if i != y_true[j] and i != y_pred[j]:
                sumstn += 1
                pass
            pass
        TP.append(sums)
        FP.append(y_pred.count(i) - sums)
        FN.append(y_true.count(i) - sums)
        TN.append(sumstn)
        pass
    allR = sum(TP) / (sum(TP) + sum(FN))
    allP = sum(TP) / (sum(TP) + sum(FP))
    if allR == 0 and allP == 0:
        print('Error: An error occurred in calculating the F1 score for multiple classifications, with the denominator being 0!')
        pass
    else:
        f1_score = 2 * allR * allP / (allR + allP)
        pass
    return f1_score



def GENuf1(y_true, y_pred):
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return macro_f1



def GENrecall_score(y_true, y_pred):
    recall_scoreNone = recall_score(y_true, y_pred, average='micro')
    return recall_scoreNone



def GENuar(y_true, y_pred):
    recall_scoremacro = recall_score(y_true, y_pred, average='macro')
    return recall_scoremacro


def GENParamtersSizeFlops(model, x, config):

    Flopsfile = os.path.join(config["paths"]["modelresult"],
                             "ParametersNumberFlops" + f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}" + ".txt")
    ms = ModelStat(model, x)
    collected_nodes = ms._analyze_model()
    report = report_format(collected_nodes)
    with open(Flopsfile, "w+") as f:
        print("[Number of model parameters and Flops]:\n", report, file=f)

    pass


def figFeascatter(feas, outputy, targes):
    figFeafiles = r'./result/feas_y_target.txt'
    figFeaf = open(figFeafiles, 'a+', encoding='utf-8')
    sum0 = 0
    sum1 = 0
    sum2 = 0
    for i in range(targes.size(0)):
        if targes[i] == 0:
            sum0 += 1
            pass
        if targes[i] == 1:
            sum1 += 1
            pass
        if targes[i] == 2:
            sum2 += 1
            pass
        print(f"features{targes[i]}={feas[i].data.cpu().numpy().tolist()}", file=figFeaf)

        pass
    _, pred = outputy.topk(1, 1, True, True)
    print(f"Outputy={outputy.data.cpu().numpy().tolist()}", file=figFeaf)
    print(f"Predictlabel={pred.data.cpu().numpy().tolist()}", file=figFeaf)
    print(f"Samplelabel={targes.data.cpu().numpy().tolist()}", file=figFeaf)
    print(f"label0_sum={sum0}.", file=figFeaf)
    print(f"label1_sum={sum1}.", file=figFeaf)
    print(f"label2_sum={sum2}.", file=figFeaf)
    figFeaf.close()

    pass



def averprCurve(true_Pred_Probas, otherparas, dpivalue=600):
    config = {
        "font.family": 'Times New Roman',
        "font.size": 9,
        "mathtext.fontset": 'stix',
        "font.weight": 'bold',
        "figure.dpi": dpivalue
    }

    configxylabel = {
        "family": 'Times New Roman',
        "size": 9,
        "weight": 'bold'
    }
    rcParams.update(config)
    dataname = ['CAS(ME)3']
    labelname = ['*']
    colorname = ['r']
    for j in range(len(true_Pred_Probas)):
        listtrue0 = []
        listtrue1 = []
        listtrue2 = []
        listtrue3 = []
        subPredProbas = true_Pred_Probas[j]
        y_score = []
        y_score.extend(subPredProbas[:, 2].tolist())
        y_score.extend(subPredProbas[:, 3].tolist())
        y_score.extend(subPredProbas[:, 4].tolist())
        y_score.extend(subPredProbas[:, 5].tolist())
        for i in range(subPredProbas.shape[0]):

            if subPredProbas[i][0] == 0:
                listtrue0.append(1)
                pass
            if subPredProbas[i][0] != 0:
                listtrue0.append(0)
                pass

            if subPredProbas[i][0] == 1:
                listtrue1.append(1)
                pass
            if subPredProbas[i][0] != 1:
                listtrue1.append(0)
                pass

            if subPredProbas[i][0] == 2:
                listtrue2.append(1)
                pass
            if subPredProbas[i][0] != 2:
                listtrue2.append(0)
                pass

            if subPredProbas[i][0] == 3:
                listtrue3.append(1)
                pass
            if subPredProbas[i][0] != 3:
                listtrue3.append(0)
                pass
            pass

        y_test = [*listtrue0, *listtrue1, *listtrue2, *listtrue3]
        averprecision, averrecall, _ = precision_recall_curve(y_test, y_score)


        aver_area = average_precision_score(y_test, y_score)
        plt.plot(averrecall, averprecision, colorname[j] + "-" + labelname[j], markersize=1, lw=1,
                 label=dataname[j] + f"_PR(AUC={aver_area:.4f})")
        pass
    plt.legend(loc='lower right')
    plt.xlabel('Recall', fontdict=configxylabel)
    plt.ylabel('Precision', fontdict=configxylabel)
    plt.title('4C-PR-Curve', fontdict=configxylabel)
    plt.grid(c='k', ls='-', lw=0.4)

    confMfigtif = os.path.join(otherparas[0] + 'PR_' + '4C' + ".tif")
    confMfigpng = os.path.join(otherparas[0] + 'PR_' + '4C' + ".png")
    confMfigpdf = os.path.join(otherparas[0] + 'PR_' + '4C' + ".pdf")
    confMfigeps = os.path.join(otherparas[0] + 'PR_' + '4C' + ".eps")
    plt.savefig(confMfigtif, dpi=dpivalue, format="tif")
    plt.savefig(confMfigpng, dpi=dpivalue, format="png")
    plt.close()

    pass


# Plot ROC curve
def averrocCurve(true_Pred_Probas, otherparas, dpivalue=600):
    config = {
        "font.family": 'Times New Roman',
        "font.size": 9,
        "mathtext.fontset": 'stix',
        "font.weight": 'bold',
        "figure.dpi": dpivalue
    }

    configxylabel = {
        "family": 'Times New Roman',
        "size": 9,
        "weight": 'bold'
    }
    rcParams.update(config)
    dataname = ['CAS(ME)3']
    labelname = ['*']
    colorname = ['r']
    for j in range(len(true_Pred_Probas)):
        listtrue0 = []
        listtrue1 = []
        listtrue2 = []
        listtrue3 = []
        subPredProbas = true_Pred_Probas[j]
        y_score = []
        y_score.extend(subPredProbas[:, 2].tolist())
        y_score.extend(subPredProbas[:, 3].tolist())
        y_score.extend(subPredProbas[:, 4].tolist())
        y_score.extend(subPredProbas[:, 5].tolist())
        for i in range(subPredProbas.shape[0]):
            if subPredProbas[i][0] == 0:
                listtrue0.append(1)
                pass
            if subPredProbas[i][0] != 0:
                listtrue0.append(0)
                pass
            if subPredProbas[i][0] == 1:
                listtrue1.append(1)
                pass
            if subPredProbas[i][0] != 1:
                listtrue1.append(0)
                pass
            if subPredProbas[i][0] == 2:
                listtrue2.append(1)
                pass
            if subPredProbas[i][0] != 2:
                listtrue2.append(0)
                pass
            if subPredProbas[i][0] == 3:
                listtrue3.append(1)
                pass
            if subPredProbas[i][0] != 3:
                listtrue3.append(0)
                pass
            pass

        y_test = [*listtrue0, *listtrue1, *listtrue2, *listtrue3]
        # averprecision, averrecall, _ = precision_recall_curve(y_test, y_score)
        # aver_area = average_precision_score(y_test, y_score)
        # print(f'y_score={y_score[-10:]}')
        # print(f'y_test={y_test[-10:]}')
        averfpr, avertpr, _ = roc_curve(y_test, y_score)
        # print(f'avertpr={avertpr[-10:]}')
        # print(f'averfpr={averfpr[-10:]}')
        averroc_auc = auc(averfpr, avertpr)
        plt.plot(averfpr, avertpr, colorname[j] + "-" + labelname[j], markersize=1, lw=1,
                 label=dataname[j] + f"_ROC(AUC={averroc_auc:.4f})")

        pass
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate', fontdict=configxylabel)
    plt.ylabel('True Positive Rate', fontdict=configxylabel)
    plt.title('4C-ROC-Curve', fontdict=configxylabel)
    plt.grid(c='k', ls='-', lw=0.4)

    confMfigtif = os.path.join(otherparas[0] + 'ROC_' + '4C' + ".tif")
    confMfigpng = os.path.join(otherparas[0] + 'ROC_' + '4C' + ".png")
    confMfigpdf = os.path.join(otherparas[0] + 'ROC_' + '4C' + ".pdf")
    confMfigeps = os.path.join(otherparas[0] + 'ROC_' + '4C' + ".eps")
    plt.savefig(confMfigtif, dpi=dpivalue, format="tif")
    plt.savefig(confMfigpng, dpi=dpivalue, format="png")
    plt.savefig(confMfigpdf, dpi=dpivalue, format="pdf")
    plt.savefig(confMfigeps, dpi=dpivalue, format="eps")
    plt.close()

    pass
