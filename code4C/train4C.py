import os
import shutil
import time, datetime
import errno
import math, random
import cv2, copy
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torchvision import transforms
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from torch.optim import lr_scheduler

import config
from models.modelNet import modelNet
from models.STSTNet import STSTNet
from getdata import *
from config import args
import resultAnalysis
from last_test import finalltest
from preprocess import deletefiles

modelpath = r'./result/modelparameters.txt'
fdme2path = r'./result/modelresult.txt'
alldatapath = r'./result/dataresult.txt'
# alldatafile = open(alldatapath, 'a+')
# fdme2 = open(fdme2path, 'a+')

losok = 0
# Perform 4 classification
class_num = 4


def datasplit(fold, paths):
    if args.sde_cde3C == 'sde4C':
        datasplitpath = 'label4Cn1.txt'
        pass
    f = open(os.path.join(paths, datasplitpath).replace('\\', '/'))
    alltrains = []
    imgdat = f.readlines()
    for i in range(len(imgdat)):
        if imgdat[i][0:imgdat[i].find('/')] != fold:
            alltrains.append(imgdat[i])
        pass
    f.close()
    train_size = int(len(alltrains) * 0.8)
    val_size = len(alltrains) - train_size
    outvalsets = []

    # trainsets, valsets = torch.utils.data.random_split(alltrains, [train_size, val_size])

    # for i in trainsets.indices:
    #     outtrainsets.append(trainsets.dataset[i])
    #     pass
    # for j in valsets.indices:
    #     outvalsets.append(valsets.dataset[j])
    #     pass
    outtrainsets = alltrains

    return outtrainsets, outvalsets


device = torch.device(args.gpunum if torch.cuda.is_available() else 'cpu')


# each fold of training and validation
def train_fold(fold, trafo_oth=None):
    alldatafile = trafo_oth[2]
    fdme2 = trafo_oth[3]
    savemodel = r'./result/modelpth'
    if not os.path.exists(savemodel):
        os.makedirs(savemodel)
    trainlosspath = r'./result'
    global best_prec1
    best_prec1 = 0

    global losok

    global val_acc
    val_acc = []

    global f1s, accs, uf1s, uars
    kwargs = {'num_workers': 0, 'pin_memory': True}

    # CASME3 Normalized (Multimodal)
    if args.sde_cde3C == 'sde4C':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [67.44569946, 75.65099565, 75.28079819]],
                                         std=[x / 255.0 for x in [21.13942046, 19.46179206, 19.71228527]])
        pass

    # whether use data augment on the image
    if args.is_augment == True:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((144, 144)),
            transforms.CenterCrop((120, 120)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            normalize,
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        pass
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    # Dataset path
    if args.sde_cde3C == 'sde4C':
        datapaths = r'../dataset4C/'
        pass

    traind, vald = datasplit(fold, datapaths)

    # whether use data split by random_split
    Experimental_Protocol = "LOSO"
    if Experimental_Protocol == "LOSO":
        if args.is_split:
            trainset = GetData_split(datapaths + fold + '/train/', 'trn_label.txt', transform_train, 4)
            # valset = GetData_split(datapaths + fold + '/train', 'val_label.txt', transform_train, 4)
            # testset = GetData_split(datapaths + fold + '/test', 'comp3C_label.txt', transform_test, 4)
        else:
            # trainset = GetData_newrawloso2(traind, fold, transform_train, 4, foldid=2)
            # testset = GetData_newrawloso2(datapaths, fold, transform_test, 4, foldid=0)
            trainset = GetData_newrawloso2(traind, fold, transform_train, 6, foldid=2)
            testset = GetData_newrawloso2(datapaths, fold, transform_test, 6, foldid=0)
            pass
    # get the train,val,test set
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=trafo_oth[4][4], shuffle=args.train_shuffle,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.valtestbatch_size, shuffle=False, **kwargs)

    if args.ispretrained:
        print(f'****model.initdata= {trafo_oth[0]}')
        # myModel = torch.load(trafo_oth[0]).to(device)
        all_part = 0
        # 【】Method 1
        if all_part == 0:
            myModel = modelNet(inputfeatues=trafo_oth[4][5], otherP=[trafo_oth[4][6], trafo_oth[4][7]]).to(device)
            myModel.load_state_dict(torch.load(trafo_oth[0]), strict=False)
            pass
        if all_part == 1:
            # 【】Method 2 load some layers
            myModel = modelNet(inputfeatues=trafo_oth[4][5], otherP=[trafo_oth[4][6], trafo_oth[4][7]]).to(device)
            pretrained_dict = torch.load(trafo_oth[0])
            model_dict = myModel.state_dict()
            layers_to_load = ['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'
                , 'fc1.weight', 'fc1.bias', 'fc3.weight', 'fc3.bias'
                , 'con1d.weight', 'con1d.bias', 'fc2.weight', 'fc2.bias'
                , 'con2d.weight', 'con2d.bias', 'conv4.weight', 'conv4.bias'
                , 'fc4.weight', 'fc4.bias']
            filtered_dict = {k: v for k, v in pretrained_dict.items()
                             if k in model_dict and k in layers_to_load}
            model_dict.update(filtered_dict)
            myModel.load_state_dict(model_dict, strict=False)
            pass
        pass
    else:
        if args.model == 'mymodel':
            myModel = modelNet(inputfeatues=trafo_oth[4][5], otherP=[trafo_oth[4][6], trafo_oth[4][7]]).to(device)
            pass
        pass

    if args.issavewbinit:
        folderlabel = os.path.basename(os.getcwd())
        saveinitmodel = '../initmodel/initmodel4C' + folderlabel[-2:]
        if not os.path.exists(saveinitmodel):
            os.mkdir(saveinitmodel)
            pass
        torch.save(myModel.state_dict(), os.path.join(saveinitmodel, 'mymodel_' + fold + '_init' + '.pth'))

        pass

    ce_criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

    if args.isallparameters:
        # optimizer SGD
        optimizer = torch.optim.SGD([{'params': myModel.parameters()}],
                                    lr=args.initial_learning_rate,
                                    momentum=args.momentum,
                                    nesterov=args.nesterov,
                                    weight_decay=args.weight_decay)
        pass
    else:

        # Frozen emotional layer, only training ECG
        frozenlists = ['conv1', 'conv2', 'fc1', 'fc3', 'maxpool', 'con1d', 'fc2', 'con2d', 'conv4', 'fc4']

        for name, param in myModel.named_parameters():
            for kf in frozenlists:
                if kf in name:
                    param.requires_grad = False
                    pass
                pass
            pass

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, myModel.parameters()),
                                    lr=args.initial_learning_rate,
                                    momentum=args.momentum,
                                    nesterov=args.nesterov,
                                    weight_decay=args.weight_decay)
        pass

    if args.lr_strategy == 0:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    if args.lr_strategy == 1:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer)
    if args.lr_strategy == 2:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    trainloss = []
    epochx = []
    allaccc = []
    uf1S = []
    uarS = []
    lrall = []
    testresult = []

    valuar = []
    valuf1 = []
    valaccs = []
    testuar = []
    testuf1 = []
    testaccs = []
    valloss = []

    losok += 1
    best_uf1 = 0
    best_uar = 0
    best_valloss = 0
    bestmetrics = []
    u1 = trafo_oth[4][0]
    u2 = trafo_oth[4][1]
    u3 = trafo_oth[4][2]
    u4 = trafo_oth[4][3]
    classws = [u1, u2, u3, u4]
    # classws = [1.0, 1.0, 1.0]
    for epoch in range(args.Epoch):

        train_lossF = nn.CrossEntropyLoss(weight=torch.tensor(classws).to(device), reduction='mean').to(device)

        adjust_learning_rate(optimizer, epoch + 1)
        # scheduler.step()
        lrall.append(optimizer.param_groups[0]['lr'])
        global EPOCH
        EPOCH = epoch
        # train for one epoch

        epochloss, acc0, umetrics, classpredtrue = train(train_loader, myModel, optimizer, epoch, train_lossF,
                                                         fold, train_P=trafo_oth[4][7])

        test_prec, test_conf, testacc0, testumetrics = test(test_loader, myModel, ce_criterion, epoch)

        is_best = testacc0 >= best_prec1
        best_prec1 = max(testacc0, best_prec1)
        best_uf1 = max(testumetrics[0][0], best_uf1)
        best_uar = max(testumetrics[0][1], best_uar)
        best_valloss = min(testumetrics[0][4], best_valloss)

        trainloss.append(epochloss)
        epochx.append(epoch)

        allaccc.append(acc0)
        uf1S.append(umetrics[0][0])
        uarS.append(umetrics[0][1])

        valaccs.append(testacc0)
        valuf1.append(testumetrics[0][0])
        valuar.append(testumetrics[0][1])
        valloss.append(float(testumetrics[0][4]))

        if is_best == True:
            nums = 0
            testaccs.append(testacc0)
            testuf1.append(testumetrics[0][0])
            testuar.append(testumetrics[0][1])
            bestmetrics.append(
                [float(f'{testacc0:.6f}'), float(f'{testumetrics[0][0]:.6f}'), float(f'{testumetrics[0][1]:.6f}'),
                 testumetrics[0][4]])

            testresult.append(
                f"{fold}_epoch{epoch}: testACC={testacc0:.6f},  testUF1={testumetrics[0][0]:.6f},  testUAR={testumetrics[0][1]:.6f} ; testLoss={testumetrics[0][4]:.6f}")
            for i in range(len(testumetrics[0][3])):
                if testumetrics[0][3][i] == testumetrics[0][2][i]:
                    nums += 1
                    pass
                pass
            testresult.append(f'y_true={testumetrics[0][3]}')
            testresult.append(f'y_pred={testumetrics[0][2]},right_num={nums}')
            torch.save(myModel.state_dict(), os.path.join(savemodel, 'mymodel_' + fold + "epoch" + str(epoch) + '.pth'))

            pass
        pass
    trainloss2 = []
    trainacc = []
    trainuf1 = []
    trainuar = []
    savelrall = []
    trainloss2.extend([round(i, 4) for i in trainloss])
    trainacc.extend([round(i, 4) for i in allaccc])
    trainuf1.extend([round(i, 4) for i in uf1S])
    trainuar.extend([round(i, 4) for i in uarS])
    savelrall.extend([round(i, 8) for i in lrall])

    valacc2 = []
    valuf12 = []
    valuar2 = []
    valacc2.extend([round(i, 4) for i in valaccs])
    valuf12.extend([round(i, 4) for i in valuf1])
    valuar2.extend([round(i, 4) for i in valuar])

    testacc2 = []
    testuf12 = []
    testuar2 = []
    testacc2.extend([round(i, 4) for i in testaccs])
    testuf12.extend([round(i, 4) for i in testuf1])
    testuar2.extend([round(i, 4) for i in testuar])

    print(f"LOSO={losok},  EPOCH={args.Epoch}, testsample：【{fold}】", file=fdme2)

    print("aver_trainloss[last10]=", np.mean(trainloss2[-10:]), file=fdme2)
    print("aver_trainacc[last10]=", np.mean(trainacc[-10:]), file=fdme2)

    print('\n', "aver_testacc[last10]=", np.mean(valacc2[-10:]), file=fdme2)

    print(f"******Test result :", file=fdme2)
    for i in testresult:
        print(i, file=fdme2)
        pass
    # ******Optimal model selection start
    allmodellist = np.array(bestmetrics)
    maxacc = np.max(allmodellist[:, 0])
    bestacclist = []
    for i in range(len(bestmetrics)):
        if maxacc == bestmetrics[i][0]:
            bestacclist.append(bestmetrics[i])
            pass
        pass
    bestacclists = np.array(bestacclist)
    maxuf1 = np.max(bestacclists[:, 1])
    maxuar = np.max(bestacclists[:, 2])
    minloss = np.min(bestacclists[:, 3])

    bestacclist0 = []
    for k in range(0, len(testresult), 3):
        topacc = float(testresult[k][testresult[k].find('=', testresult[k].find(':')) + 1:
                                     testresult[k].find(',')])
        if maxacc == topacc:
            bestacclist0.append(testresult[k])
            pass
        pass

    bestacclist = bestacclist0
    list2 = []
    list3 = []
    for k in range(len(bestacclist)):
        topuf1 = float(bestacclist[k][bestacclist[k].find('testUF1') + len('testUF1='):
                                      bestacclist[k].find(',', bestacclist[k].find('testUF1')) - 1])
        topuar = float(bestacclist[k][bestacclist[k].find('testUAR') + len('testUAR='):
                                      bestacclist[k].find(';') - 1])
        toploss = float(bestacclist[k][bestacclist[k].find('=', bestacclist[k].find(';')) + 1:])
        if topuf1 == maxuf1 and topuar == maxuar and toploss == minloss:
            list3.append(bestacclist[k])
            pass
        if topuf1 == maxuf1 and topuar == maxuar and toploss != minloss:
            list2.append(bestacclist[k])
            pass
        pass
    if len(list3) != 0:
        toptestresult = list3[0]
        pass
    if len(list3) == 0 and len(list2) != 0:
        temploss = []
        for k in range(len(list2)):
            toploss = float(list2[k][list2[k].find('=', list2[k].find(';')) + 1:])
            temploss.append(toploss)
            pass
        tempminloss = np.min(np.array(temploss))
        for k in range(len(list2)):
            toploss = float(list2[k][list2[k].find('=', list2[k].find(';')) + 1:])
            if tempminloss == toploss:
                toptestresult = list2[k]
                pass
            pass
        pass
    if len(list3) == 0 and len(list2) == 0:
        for k in range(len(bestacclist)):
            toploss = float(bestacclist[k][bestacclist[k].find('=', bestacclist[k].find(';')) + 1:])
            if toploss == minloss:
                toptestresult = bestacclist[k]
                pass
            pass
        pass

    trainsavemodel = os.path.join(trainlosspath, 'trainresult').replace('\\', '/')
    testsavemodel = os.path.join(trainlosspath, 'testresult').replace('\\', '/')
    if not os.path.exists(trainsavemodel):
        os.makedirs(trainsavemodel)
    if not os.path.exists(testsavemodel):
        os.makedirs(testsavemodel)

    resultAnalysis.GENxycurve(x=epochx, y=trainloss, savepath=trainsavemodel, index=losok,
                              imgnname=fold + "trainLoss")
    resultAnalysis.GENxycurve(x=epochx, y=trainacc, savepath=trainsavemodel, index=losok,
                              imgnname=fold + "trainAcc")

    resultAnalysis.GENxycurve(x=epochx, y=valacc2, savepath=testsavemodel, index=losok,
                              imgnname=fold + "testAcc")
    resultAnalysis.GENxycurve(x=epochx, y=valloss, savepath=testsavemodel, index=losok,
                              imgnname=fold + "testLoss")
    print(f'【LOSO ,testsub={fold}】', file=alldatafile)
    print(f'【epochx】=  {epochx}', file=alldatafile)
    print(f'【trainLoss】=  {trainloss2}', file=alldatafile)
    print(f'【trainAcc】=  {trainacc}', file=alldatafile)

    print(f'\n【testAcc】=  {valacc2}', file=alldatafile)
    # print(f'【valUF1】=  {valuf12}', file=alldatafile)
    print(f'【testLoss】=  {valloss}', file=alldatafile)
    print(f'【LR】=  {savelrall}', file=alldatafile)

    if fold == '01' or fold == '006':
        resultAnalysis.GENxycurve(x=epochx, y=lrall, savepath=trainlosspath, index=losok, imgnname="lr")
        pass
    deletemodels(bestmodel=toptestresult, folds=fold)
    return toptestresult


def deletemodels(bestmodel, folds):
    temps = r'./result/modelpth'
    allmodels = os.listdir(temps)
    for i in range(len(allmodels)):
        if folds == allmodels[i][allmodels[i].find('_') + 1:allmodels[i].find('epoch')]:
            if bestmodel[bestmodel.find('epoch'):bestmodel.find(':')] != allmodels[i][
                                                                         allmodels[i].find('epoch'): allmodels[
                                                                             i].find('.pth')]:
                deltpath = os.path.join(temps, allmodels[i]).replace('\\', '/')
                os.remove(deltpath)
                pass
            pass
        pass
    pass


def normal_func(x, sigma=0.39894, norm01=False):
    y0 = []
    for x0 in x:
        x0 = x0 / 10
        y = pow(math.e, (-1.0 * pow(float(x0), 2) / (2 * pow(sigma, 2)))) / (pow(2 * math.pi, 0.5) * sigma)
        y0.append(y)
        pass
    minv = min(y0)
    maxv = max(y0)
    if norm01:
        # 这是映射到0-1之间
        y00 = [(i - minv) / (maxv - minv) for i in y0]
        pass
    else:
        y00 = y0
        pass
    return y00


def active_balence(prednum, truenum):
    labelration = np.sum(np.array(prednum) == np.array(truenum)) / len(truenum)
    classnum = []
    classrightnum = []
    for i in list(set(truenum)):
        nums = 0
        classum = 0
        for j in range(len(truenum)):
            if i == prednum[j] and i == truenum[j]:
                nums += 1
                pass
            if i == truenum[j]:
                classum += 1
                pass
            pass
        classrightnum.append(nums)
        classnum.append(classum)
        pass
    classratio = np.array(classrightnum) / np.array(classnum)
    deltaw = labelration - classratio
    # classw=1 - deltaw
    classw = normal_func(deltaw)

    return classw


def train(train_loader, trainModel, optimizer, epoch, ce_Wcriterion, fold, train_P=None):
    losses = AverageMeter()
    trainModel.train()
    epochy_trues = []
    epochy_preds = []
    umetric = []
    balancelabels = []
    tq = tqdm(train_loader, leave=True, ncols=100)
    batchs = 0
    iteration = 0
    sumbatch = 0
    sumtrue0 = 0
    sumtrue1 = 0
    sumtrue2 = 0
    sumtrue3 = 0
    sumtar0 = 0
    sumtar1 = 0
    sumtar2 = 0
    sumtar3 = 0
    for i, (x1, x2, x3, x4, x5, x6, target) in enumerate(tq):
        iteration += 1
        torch.set_printoptions(precision=8, sci_mode=False)
        tq.set_description(f"test_subs={fold}")
        balancelabels.extend(target.data.tolist())
        batchs += 1
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        x4 = x4.to(device)
        x5 = x5.to(device)
        x6 = x6.to(device)
        target = target.to(device)
        input_var1 = torch.autograd.Variable(x1)
        input_var2 = torch.autograd.Variable(x2)
        input_var3 = torch.autograd.Variable(x3)
        input_var4 = torch.autograd.Variable(x4)
        input_var5 = torch.autograd.Variable(x5)
        input_var6 = torch.autograd.Variable(x6)
        target_var = torch.autograd.Variable(target)
        output, outputxy = trainModel(input_var1, input_var2, input_var3, input_var4, input_var5, input_var6,
                                      target_var)
        currbatch = x1.size(0)
        sumbatch = sumbatch + currbatch
        _, pred = output.topk(1, 1, True, True)
        _, predxy = outputxy.topk(1, 1, True, True)
        true0 = 0
        true1 = 0
        true2 = 0
        true3 = 0
        tar0 = 0
        tar1 = 0
        tar2 = 0
        tar3 = 0
        for kk in range(len(target)):
            if target[kk] == 0:
                tar0 += 1
                pass
            if target[kk] == 1:
                tar1 += 1
                pass
            if target[kk] == 2:
                tar2 += 1
                pass
            if target[kk] == 3:
                tar3 += 1
                pass
            pass
        sumtar0 += tar0
        sumtar1 += tar1
        sumtar2 += tar2
        sumtar3 += tar3

        for jj in range(len(pred)):
            if pred[jj] == target[jj] and pred[jj] == 0:
                true0 += 1
                pass
            if pred[jj] == target[jj] and pred[jj] == 1:
                true1 += 1
                pass
            if pred[jj] == target[jj] and pred[jj] == 2:
                true2 += 1
                pass
            if pred[jj] == target[jj] and pred[jj] == 3:
                true3 += 1
                pass
            pass
        sumtrue0 += true0
        sumtrue1 += true1
        sumtrue2 += true2
        sumtrue3 += true3
        p0 = 0
        p1 = 0
        p2 = 0
        p3 = 0
        if sumtar0 != 0:
            p0 = sumtrue0 / sumtar0
            pass
        if sumtar1 != 0:
            p1 = sumtrue1 / sumtar1
            pass
        if sumtar2 != 0:
            p2 = sumtrue2 / sumtar2
            pass
        if sumtar3 != 0:
            p3 = sumtrue3 / sumtar3
            pass
        pis = [p0, p1, p2, p3]

        loss = lossFunc(output, outputxy, target_var, losscoeff=train_P, lf_othersp=pis)
        epochy_trues.extend(target.data.cpu().numpy().tolist())
        epochy_preds.extend(pred.view(1, -1).cpu().numpy().tolist()[0])

        losses.update(loss.data.item(), x2.size(0))
        # *************************************
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pass

    f1s, accs, uf1s, uars = modelmetric(epochy_preds, epochy_trues)
    umetric.append([uf1s, uars, epochy_preds, epochy_trues])
    classpredtrue = [epochy_preds, epochy_trues]
    return losses.ave, accs, umetric, classpredtrue


# Custom loss function
def lossFunc(outputs, outputsxy, targets, losscoeff=1.0, lf_othersp=None):
    classN = class_num
    softmaxs = nn.Softmax(dim=1)
    a = softmaxs(torch.tensor(lf_othersp).view(1, classN)).to(device)
    wi = (1 - a) / (classN - 1)
    logsoftmaxs = nn.LogSoftmax(dim=1)
    logsoftmaxp = logsoftmaxs(outputs)

    labelp = torch.zeros(logsoftmaxp.size(0))
    demos = []
    for i in range(logsoftmaxp.size(0)):
        labelp[i] = -1.0 * wi[0][targets[i]] * logsoftmaxp[i][targets[i]]
        demos.append(wi[0][targets[i]])
        pass
    lossv = torch.sum(labelp) / sum(demos)

    return lossv


def test(test_loader, testmodel, criterion, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    testmodel.eval()
    testepochy_trues = []
    testepochy_preds = []
    testumetric = []

    # end = time.time()
    conf_matrix = torch.zeros(3, 3)
    for i, (x1, x2, x3, x4, x5, x6, target) in enumerate(test_loader):
        target = target.to(device)
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        x4 = x4.to(device)

        x5 = x5.to(device)
        x6 = x6.to(device)
        input_var1 = torch.autograd.Variable(x1)
        input_var2 = torch.autograd.Variable(x2)
        input_var3 = torch.autograd.Variable(x3)
        input_var4 = torch.autograd.Variable(x4)
        input_var5 = torch.autograd.Variable(x5)
        input_var6 = torch.autograd.Variable(x6)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            # output = testmodel(input_var1, input_var2, input_var3, input_var4, target_var)
            output, _ = testmodel(input_var1, input_var2, input_var3, input_var4, input_var5, input_var6, target_var)
            pass

        loss = criterion(output, target_var)
        prec1 = accuracy(output.data, target, topk=(1,))[0]

        losses.update(loss.data.item(), x2.size(0))
        top1.update(prec1.item(), x2.size(0))

        _, pred = output.topk(1, 1, True, True)
        testepochy_trues.extend(target.data.cpu().numpy().tolist())
        testepochy_preds.extend(pred.view(1, -1).cpu().numpy().tolist()[0])
        pass
    f1s, testacc, testuf1s, testuars = modelmetric(testepochy_preds, testepochy_trues)
    testumetric.append([testuf1s, testuars, testepochy_preds, testepochy_trues, float(f'{losses.ave:.6f}')])

    val_acc.append(top1.ave)

    return top1.ave, conf_matrix.numpy(), testacc, testumetric


class Full_layer(torch.nn.Module):
    def __init__(self, feature_num, class_num):
        super(Full_layer, self).__init__()
        self.class_num = class_num
        self.fc = nn.Linear(feature_num, class_num)

    def forward(self, x):
        x = self.fc(x)
        return x


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count


def adjust_Balance_W(epoch, label):
    classnum = []
    balanceratio = []
    mu = []
    epsilon = 0.002
    classnum.extend([label.count(i) for i in range(3)])
    balanceratio.extend([j / sum(classnum) + epsilon for j in classnum])
    mu.append(1.0 / (1 + balanceratio[0] / balanceratio[1] + balanceratio[0] / balanceratio[2]))
    mu.append(mu[0] * balanceratio[0] / balanceratio[1])
    mu.append(mu[0] * balanceratio[0] / balanceratio[2])

    return mu


def adjust_learning_rate(optimizer, epoch):
    if args.changelr == 12:
        threadvalue12 = 135
        if epoch < threadvalue12:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.01
                pass
            pass
        if threadvalue12 <= epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-3
                pass
            pass
        pass
    pass


def modelmetric(output, target):
    y_true = target
    y_pred = output
    f1 = resultAnalysis.GENf1_score(y_true, y_pred)
    acc = resultAnalysis.GENaccuracy_score(y_true, y_pred)
    uf1, uar = resultAnalysis.recognition_evaluation(y_true, y_pred)

    return f1, acc, uf1, uar


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


def computefinalresult(finalresults):
    mid = 3
    alltestmetris = np.zeros((len(finalresults), mid))
    allvalmetris = np.zeros((len(finalresults), 4))
    for k in range(len(finalresults)):
        testresus = simpletop(finalresults[k])
        alltestmetris[k][0] = testresus[0]
        alltestmetris[k][1] = testresus[1]
        alltestmetris[k][2] = testresus[2]
        pass
    return allvalmetris, alltestmetris


def simpletop(topfinalresult):
    tacci = topfinalresult.find('testACC=')
    tuf1i = topfinalresult.find('testUF1=')
    tuari = topfinalresult.find('testUAR=')
    finalacc = topfinalresult[tacci + len('testACC='):tacci + len('testACC=') + 6]
    finaluf1 = topfinalresult[tuf1i + len('testUF1='):tuf1i + len('testUF1=') + 6]
    finaluar = topfinalresult[tuari + len('testUAR='):tuari + len('testUAR=') + 6]
    testresus = [float(finalacc.strip()), float(finaluf1.strip()), float(finaluar.strip())]
    return testresus


def mains(othersP=None):
    alldatafile = open(alldatapath, 'a+', encoding='utf-8')
    fdme2 = open(fdme2path, 'a+', encoding='utf-8')

    finaltopresults = []

    startime = f"{datetime.datetime.now():%Y.%m.%d.%H.%M.%S}"
    startt = time.time()

    if args.sde_cde3C == 'sde4C':
        labelpath = r'../dataset4C/label4Cn1.txt'
        pass
    with open(labelpath, 'r+') as f:
        files = f.readlines()
        pass
    folds = []
    for name in files:
        if name[0:name.find('/')] in folds:
            continue
        folds.append(name[0:name.find('/')])
        pass
    i = 0

    if args.ispretrained:

        inittrain = 'Custom initialization path'
        initpath = r'../pretrained' + '/' + inittrain
        modelsluist = os.listdir(initpath)

        pass
    else:
        initpath = f'Random initialization……'
        print(f'############【Random initialization……】')
        pass
    for fold in folds:
        print(f'********++++++{i + 1}/{len(folds)}')
        record_path = args.save_path + '/' + fold
        if fold != 'total' and not os.path.exists(record_path):
            if args.ispretrained:
                for k in range(len(modelsluist)):
                    if inittrain[-4:] == 'best':
                        modelsub = modelsluist[k][modelsluist[k].find('_') + 1:modelsluist[k].find('epoch')]
                        if fold == modelsub:
                            cnnmodel = os.path.join(initpath, modelsluist[k]).replace('\\', '/')
                            pass
                        pass
                    else:
                        modelsub = modelsluist[k][modelsluist[k].find('_') + 1:modelsluist[k].find('_init')]
                        if fold == modelsub:
                            cnnmodel = os.path.join(initpath, modelsluist[k]).replace('\\', '/')
                        pass
                        pass
                    pass
                pass
            else:
                cnnmodel = None
                pass
            othmodel = None

            i += 1
            topfinalresult = train_fold(fold, trafo_oth=[cnnmodel, othmodel, alldatafile, fdme2, othersP])
            finaltopresults.append(topfinalresult)
            pass
        pass
    allvalmetris, alltestmetris = computefinalresult(finaltopresults)

    print(f'\n////////////////////【bestStart】///////////////////////////', file=fdme2)
    for i in finaltopresults:
        print(i, file=fdme2)
        pass
    print(f'////////////////////【bestEnd】///////////////////////////', file=fdme2)

    print(
        f'【Max】testing：ACC={np.max(alltestmetris, axis=0)[0]:.4f}, UF1={np.max(alltestmetris, axis=0)[1]:.4f}, UAR={np.max(alltestmetris, axis=0)[2]:.4f}',
        file=fdme2)
    print(
        f'【Aver】testing：ACC={np.mean(alltestmetris, axis=0)[0]:.4f}, UF1={np.mean(alltestmetris, axis=0)[1]:.4f}, UAR={np.mean(alltestmetris, axis=0)[2]:.4f}',
        file=fdme2)


    endtime = f"{datetime.datetime.now():%Y.%m.%d.%H.%M.%S}"
    endt = time.time()
    print(
        f'training:\n{startime}---{endtime}\ntime:\n{endt - startt:.4f}(s) or {(endt - startt) / 60:.4f}(min) or {(endt - startt) / 3600:.4f}(h)',
        file=fdme2)
    fdme2.close()
    alldatafile.close()

    print("*************Finally, the model starts testing!!! ")
    finalltest(params=[othersP[5], othersP[6], othersP[7]])
    print(f'***** TestStage Exit Model! ')
    print("*************Finally, the model tests Done!!! ")

    pass


# ***************************************************

if __name__ == '__main__':
    startimes = f"{datetime.datetime.now():%Y/%m/%d; %H:%M:%S}"
    currentfolder = os.path.basename(os.getcwd())
    logpath = r'./result'
    if not os.path.exists(logpath):
        os.mkdir(logpath)
        pass
    initmodels = '../initmodel'
    if not os.path.exists(initmodels):
        os.mkdir(initmodels)
        pass
    deletefiles(specialpath=r'./result', flags=1)
    fdme3pathn = r'./result/gridSearchResult.txt'
    fdme3pathn2 = r'./result/gridSearchResult2.txt'
    preci = 1000

    u2 = 1.0
    u1 = 1.0
    u3 = 1.0
    u4 = 1.0
    alpha = 0.5
    otherparas = 0.87
    mains(othersP=[u1, u2, u3, u4, 128, 128, alpha, otherparas])
    currtimes = f"{datetime.datetime.now():%Y/%m/%d; %H:%M:%S}"
    print(f'###Code, start time： {startimes}')
    print(f'###Code, end time： {currtimes}')

    pass
