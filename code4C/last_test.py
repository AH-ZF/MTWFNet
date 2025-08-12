'''
This py module is mainly used to test the final performance of the model.
'''

from torchvision import transforms
import os, datetime, time
import pandas as pd

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from config import args
from getdata import GetData_newrawloso2
import resultAnalysis
from models.modelNet import modelNet
import torch.nn.functional as F
from ptflops import get_model_complexity_info
from thop import profile



device = torch.device(args.gpunum if torch.cuda.is_available() else 'cpu')


if args.sde_cde3C == 'sde4C':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [67.44569946, 75.65099565, 75.28079819]],
                                     std=[x / 255.0 for x in [21.13942046, 19.46179206, 19.71228527]])
    normalizeecgn1 = transforms.Normalize(mean=[-66.73242274274789], std=[713.2221458449152])
    pass



transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize
])
kwargs = {'num_workers': 0, 'pin_memory': True}

class_num = 4
testpath = r'./result/code/result'
modelmycnn = r'./result/modelpth/mycnnmodel.pth'
modelfc = r'./result/modelpth/fcmodel.pth'

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


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        # res.append(correct_k.mul_(100.0 / batch_size))
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


def modelmetric(output, target):

    y_true = target
    y_pred = output
    f1 = resultAnalysis.GENf1_score(y_true, y_pred)
    selff1 = resultAnalysis.self_f1(y_true, y_pred)
    acc = resultAnalysis.GENaccuracy_score(y_true, y_pred)
    uf1, uar = resultAnalysis.recognition_evaluation(y_true, y_pred)
    paaccs, paf1_score, pauf1, pauar = resultAnalysis.paper_metric(y_true, y_pred)
    papermet = [paaccs, paf1_score, pauf1, pauar]
    return f1, selff1, acc, uf1, uar, papermet


def test(test_loader, model, criterion):

    losses = AverageMeter()

    model.eval()
    testepochy_trues = []
    testepochy_preds = []
    losssub = []


    empty_array = np.empty((0, class_num))
    tq = tqdm(test_loader, leave=True, ncols=100)
    for i, (x, x2, x3, x4, x5, x6, target) in enumerate(tq):
        target = target.to(device)
        x1 = x.to(device)
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
            # output = model(input_var1, input_var2, input_var3, input_var4, target_var)
            output, _ = model(input_var1, input_var2, input_var3, input_var4, input_var5, input_var6, target_var)
            pass

        loss = criterion(output, target_var)
        losssub.append((loss.data.item()) * (x2.size(0)))
        prec1 = accuracy(output.data, target, topk=(1,))[0]

        losses.update(loss.data.item(), x2.size(0))
        _, pred = output.topk(1, 1, True, True)
        outputs = F.softmax(output, dim=1)

        empty_array = np.concatenate((empty_array, outputs.data.cpu().numpy()), axis=0)
        testepochy_trues.extend(target.data.cpu().numpy().tolist())
        testepochy_preds.extend(pred.view(1, -1).cpu().numpy().tolist()[0])
        pass
    vector_2dtrues = np.array(testepochy_trues).reshape(-1, 1)
    vector_2dpreds = np.array(testepochy_preds).reshape(-1, 1)
    truePredProbas = np.hstack((vector_2dtrues, vector_2dpreds, empty_array))

    return testepochy_preds, testepochy_trues, [losssub, truePredProbas]


def main(models, fold, otherparams):
    if args.sde_cde3C == 'sde4C':
        datapaths = r'../dataset4C/'
        pass

    testset = GetData_newrawloso2(datapaths, fold, transform_test, 6, foldid=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, **kwargs)

    print(f'***** TestStage Entry Model! ')
    model = modelNet(inputfeatues=otherparams[0], otherP=[otherparams[1], otherparams[2]]).to(device)
    pretraind_dict = torch.load(models)
    model_dict = model.state_dict()
    state_dict = {}
    for k, v in pretraind_dict.items():
        # if k in model_dict.keys() and k not in ['con1d.weight', 'con1d.bias']:
        if k in model_dict.keys():
            state_dict[k] = v
        pass
    # state_dict = {k: v for k, v in pretraind_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    # Complexity of computational models
    if fold == '01':
        model.eval()
        input1 = torch.randn(2, 3, 12, 12).to(device)
        input2 = torch.randn(2, 3, 12, 12).to(device)
        input3 = torch.randn(2, 3, 12, 12).to(device)
        input4 = torch.randn(2, 3, 12, 12).to(device)  #
        input5 = torch.randn(2, 3, 12, 12).to(device)
        input6 = torch.randn(2, 1, 1, 18).to(device)
        labelt = torch.tensor([0,2]).to(device)
        inputall = (input1, input2, input3, input4, input5, input6,labelt)
        modelComplexity = './result/model_Complexity.txt'

        complexfiles = open(modelComplexity, 'a+', encoding='utf-8')
        print(f'\n*************************************fold={fold}:', file=complexfiles)
        allps = sum([p.data.nelement() for p in model.parameters()])
        print(f'#Model parameter count Params1 (manual): {allps / 1e6}Million(M)---{allps}', file=complexfiles)



        def input_constructor(input_res):
            return {
                "x1": input1,
                "x2": input2,
                "x3": input3,
                "x4": input4,
                "x5": input5,
                "x6": input6,
                "labels": labelt,
            }

        #  get_model_complexity_info
        ptflopsmacs, ptflopsparams = get_model_complexity_info(
            model,
            input_res=(2,),
            input_constructor=input_constructor,
            as_strings=False,
            print_per_layer_stat=True,
            verbose=False
        )
        print(f'#Params4(ptflops): {ptflopsparams / 1e6}(M)---{ptflopsparams}', file=complexfiles)
        print(f'#MACs(ptflops): {ptflopsmacs / 1e6}(M)---{ptflopsmacs}', file=complexfiles)
        thopflops, thopparams = profile(model, inputall)
        print(f'#Params(thop): {thopparams / 1e6}(M)---{thopparams}', file=complexfiles)
        print(f'#FLOPs(thop): {thopflops / 1e6}(M)---{thopflops}', file=complexfiles)
        complexfiles.close()
        pass

    ce_criterion = nn.CrossEntropyLoss().to(device)

    model = model.to(device)
    testepochy_preds, testepochy_trues, losssubs = test(test_loader, model, ce_criterion)

    return testepochy_preds, testepochy_trues, losssubs


fdme2pathn = r'./result/modelresult.txt'
fdme3pathn = r'./result/gridSearchResult.txt'
fdme3pathn2 = r'./result/gridSearchResult2.txt'
modelres = r'./result/modelpth'


def newmetric():
    fdmenew = open(fdme2pathn, 'r+', encoding='utf-8')
    lines = fdmenew.readlines()
    modellist = []

    testsub = []
    modelname = os.listdir(modelres)
    for i in range(len(lines)):
        if lines[i].find('bestStart') != -1:
            stindx = i
            pass
        if lines[i].find('bestEnd') != -1:
            endindx = i
            pass
        pass
    bestmodel = lines[stindx + 1:endindx]
    for j in bestmodel:
        if len(j) < 2:
            continue
        for k in modelname:
            if j[0:j.find('_')] == k[k.find('_') + 1:k.find('epoch')] and \
                    j[j.find('epoch'):j.find(':')] == k[
                                                      k.find('epoch'):k.find(
                                                          '.')]:
                modellist.append(os.path.join(modelres, k).replace('\\', '/'))
                testsub.append(j[0:j.find('_')])
                pass

            # if k[0:2] == 'fc':
            #     if j[0:j.find('_')] == k[k.find('_') + 1:k.find('epoch')] and \
            #             j[j.find('epoch'):j.find(':')] == k[
            #                                               k.find(
            #                                                   'epoch'):k.find(
            #                                                   '.')]:
            #         fcmodellist.append(os.path.join(modelres, k).replace('\\', '/'))
            #         pass
            #     pass
            pass
        pass

    return modellist, testsub


def deletefiles(mycnnmodellists, fcmodellists):
    path = r'./result/modelpth'
    files = os.listdir(path)
    finalfiles = []
    for i in range(len(mycnnmodellists)):
        finalfiles.append(mycnnmodellists[i][mycnnmodellists[i].find('mycnn'):])
        pass
    for i in range(len(fcmodellists)):
        finalfiles.append(fcmodellists[i][fcmodellists[i].find('fcmodel'):])
        pass
    for fil in files:
        if fil not in finalfiles:
            deltpath = os.path.join(path, fil)
            os.remove(deltpath)
            pass
        pass
    pass


def testresult(modellists, testsubs, dataname, otherparams):

    y_preds = []
    y_true = []
    allloss = []
    allnums = []
    singlesub = []
    singlenum = []
    samps = 0
    alltrue_Pred_Probas = np.empty((0, class_num + 2))
    for fold in testsubs:
        samps += 1

        for i in modellists:
            if i[i.find('_') + 1:i.find('epo')] == fold:
                models = i
            pass
        print(f"test_subs={fold}")
        print(f'Best_model={models}')


        testsuby_preds, testsub_trues, mlosssubs = main(models, fold, otherparams)

        singlesub.append([fold + ': ', testsub_trues, testsuby_preds])
        nums = 0
        for i in range(len(testsub_trues)):
            if testsub_trues[i] == testsuby_preds[i]:
                nums += 1
            pass

        allnums.append(nums)

        y_preds.extend(testsuby_preds)
        singlenum.append(len(testsub_trues))
        y_true.extend(testsub_trues)
        allloss.extend(mlosssubs[0])
        alltrue_Pred_Probas = np.concatenate((alltrue_Pred_Probas, mlosssubs[1]), axis=0)
        pass
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0
    for jj in range(len(y_true)):
        if y_true[jj] == y_preds[jj] == 0:
            count0 += 1
            pass
        if y_true[jj] == y_preds[jj] == 1:
            count1 += 1
            pass
        if y_true[jj] == y_preds[jj] == 2:
            count2 += 1
            pass
        if y_true[jj] == y_preds[jj] == 3:
            count3 += 1
            pass

        pass
    print(f"Total number of correctly predicted samples={sum(allnums)}")

    print(f"Predicted category 0 numbers={count0}/{y_true.count(0)}")
    print(f"Predicted category 1 numbers={count1}/{y_true.count(1)}")
    print(f"Predicted category 2 numbers={count2}/{y_true.count(2)}")
    print(f"Predicted category 3 numbers={count3}/{y_true.count(3)}")
    print(f"Total number of correctly predicted samples={allnums}")

    f1s, selff1s, testacc, testuf1s, testuars, papermets = modelmetric(y_preds, y_true)
    finaloss = sum(allloss) / len(y_preds)

    girdf = open(fdme3pathn, 'a+', encoding='utf-8')
    girdf2 = open(fdme3pathn2, 'a+', encoding='utf-8')
    print(f'***Finally Performance :{dataname}', file=girdf)
    print(
        f'Allrightnum={sum(allnums)}/{len(y_preds)}, Acc={testacc:.6f},F1={papermets[1]:.6f}, UF1={testuf1s:.6f}, UAR={testuars:.6f}, Loss={finaloss:.6f}',
        file=girdf)
    print(
        f'Allrightnum={sum(allnums)}/{len(y_preds)}, Acc={testacc:.6f},F1={papermets[1]:.6f}, UF1={testuf1s:.6f}, UAR={testuars:.6f}, Loss={finaloss:.6f}',
        file=girdf2)
    print(f"Number of correctly predicted samples={allnums}", file=girdf)
    print(f"Total number of correctly predicted samples={sum(allnums)}", file=girdf)
    print(f"Prediction correct, category 0={count0}/{y_true.count(0)}", file=girdf)
    print(f"Prediction correct, category 1={count1}/{y_true.count(1)}", file=girdf)
    print(f"Prediction correct, category 2={count2}/{y_true.count(2)}", file=girdf)
    print(f"Prediction correct, category 3={count3}/{y_true.count(3)}", file=girdf)

    girdf.close()
    girdf2.close()

    with open(fdme2pathn, 'a+', encoding='utf-8') as testf:
        print(72 * '+', file=testf)
        print(f'********Finally Performance :{dataname}', file=testf)
        print(
            f'Allrightnum={sum(allnums)}/{len(y_preds)}, Acc={testacc:.6f},F1={papermets[1]:.6f}, UF1={testuf1s:.6f}, UAR={testuars:.6f}, Loss={finaloss:.6f}',
            file=testf)
        print(f"Number of correctly predicted samples={allnums}", file=testf)
        print(f"Total number of correctly predicted samples={sum(allnums)}/{len(y_preds)}", file=testf)
        print(f"Prediction correct, category0={count0}/{y_true.count(0)}", file=testf)
        print(f"Prediction correct, category1={count1}/{y_true.count(1)}", file=testf)
        print(f"Prediction correct, category2={count2}/{y_true.count(2)}", file=testf)
        print(f"Prediction correct, category3={count3}/{y_true.count(3)}", file=testf)
        print(f"Theoretical sample size={singlenum}", file=testf)
        print(f"Sample prediction results= \n{singlesub}", file=testf)
        print(72 * '+', file=testf)
        pass
    # df = pd.DataFrame(fin_features)
    # df1 = pd.DataFrame(fin_targets)
    # df.to_pickle('./result/alltestfeatures3C_' + dataname + '.pkl')
    # df1.to_pickle('./result/alltestlabels3C_' + dataname + '.pkl')
    resultAnalysis.GENconfusion_matrix(y_true, y_preds, savepath='./result/', index=0,
                                       imgnname='AAtestCMatrix' + dataname)
    # resultAnalysis.figuretsne(featurepath=fin_features, labelpath=fin_targets, savepath='./result/',
    #                           imgnname='t-SNE_' + dataname)

    return alltrue_Pred_Probas


def finalltest(params=[]):
    modellists, testsubs = newmetric()

    lengths = len(testsubs)
    cassubs = []
    casme3subs = []
    for j in range(lengths):
        if testsubs[j][0:3] == 'sub':
            cassubs.append(testsubs[j])
            pass
        # Only calculate casme3 (multimodal)
        casme3subs.append(testsubs[j])

    if args.sde_cde3C == 'sde4C':
        CASME3_0 = time.time()
        casme3prRoc = testresult(modellists, testsubs=casme3subs, dataname='CASME3', otherparams=params)
        CASME3_1 = time.time()
        print(f'Total time under CAS(ME)3：{CASME3_1 - CASME3_0}(s)')
        pass
        # Plot the average PR and ROC curves for all datasets
    resultAnalysis.averrocCurve([casme3prRoc], otherparas=['./result/'])
    resultAnalysis.averprCurve([casme3prRoc], otherparas=['./result/'])
    pass


if __name__ == '__main__':

    tempfiles = open(fdme2pathn, 'a+', encoding='utf-8')
    print(f"\nTesting time：【{datetime.datetime.now():%Y/%m/%d %H:%M:%S}】", file=tempfiles)

    print(f"###############Test the last_test.py file below!!!###############\n", file=tempfiles)
    tempfiles.close()
    finalltest([128, 0.5, 0.87])
    print('Test complete!!!')

    pass
