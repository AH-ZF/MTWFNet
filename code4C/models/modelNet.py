import math

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from sklearn.metrics.pairwise import cosine_similarity
from resultAnalysis import figFeascatter
import cbam
from config import args
from math import e as natureE



device = torch.device(args.gpunum if torch.cuda.is_available() else 'cpu')


class modelNet(nn.Module):
    def __init__(self, inputfeatues, otherP=None):
        super(modelNet, self).__init__()
        self.fusmodel = 1
        self.alphas = otherP[0]
        self.beta = 0.64
        self.gamma = otherP[1]
        self.flagatt = "A"
        self.feature_num = inputfeatues
        self.feature_num2 = 16
        self.class_num = 4  # 即4分类
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.fc1 = nn.Linear(in_features=576, out_features=self.feature_num)
        self.fc3 = nn.Linear(in_features=self.feature_num, out_features=self.class_num)

        self.conv4 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.fc4 = nn.Linear(in_features=288, out_features=self.feature_num)

        self.con1d = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc2 = nn.Linear(in_features=288, out_features=self.feature_num)

        self.con2d = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.fc5 = nn.Linear(in_features=27, out_features=self.feature_num)

        self.relu = nn.ReLU()
        self.dps = nn.Dropout(0.5)

        pass

    def flattenmodule(self, in1, in2, in3, in4):

        catC = torch.cat((in1, in2, in3, in4), dim=1)  #
        catT = torch.zeros_like(catC)
        for i in range(0, catT.size(1) - 4, 4):
            catT[:, i, ...] = in1[:, int(i / 4), ...]
            catT[:, i + 1, ...] = in2[:, int(i / 4), ...]
            catT[:, i + 2, ...] = in3[:, int(i / 4), ...]
            catT[:, i + 3, ...] = in4[:, int(i / 4), ...]
            pass
        catTB = catT.view(catT.size(0), catT.size(1), -1)
        catTD = catC.view(catC.size(0), catC.size(1), -1)
        catTA = torch.zeros_like(catTB)
        catTC = torch.zeros_like(catTB)
        for j in range(catT.size(2)):
            catTA[:, :, j * 6:j * 6 + 6] = catT[:, :, :, j]
            catTC[:, :, j * 6:j * 6 + 6] = catC[:, :, :, j]
            pass
        catTAC = torch.cat((catTA, catTC), dim=1)
        catTBD = torch.cat((catTB, catTD), dim=1)
        # return catTA, catTB, catTC, catTD
        return catTAC, catTBD

    def fusionmodule(self, insAC1, insBD2, oldx):

        # fusC = torch.zeros_like(oldx)
        fusBD = insBD2.view(insBD2.size(0), insBD2.size(1), 6, 6)
        fusAC = torch.zeros_like(fusBD)

        for k in range(fusAC.size(2)):
            fusAC[:, :, :, k] = insAC1[:, :, k * 6:k * 6 + 6]
            pass

        # print(f'alphas=={self.alphas}')
        xx = self.alphas * (fusAC + fusBD)

        y = self.relu(xx)
        # y = torch.sigmoid(xx)
        return y

    def attention128(self, fea1, fea2):

        maxfea = torch.zeros_like(fea1)
        for ii in range(fea1.size(0)):
            maxfea[ii] = torch.max(torch.cat((fea1[ii].view(1, -1), fea2[ii].view(1, -1)), dim=0), dim=0)[0].view(1, -1)
            pass
        meanfea = 0.5 * (fea1 + fea2)

        return maxfea, meanfea


    def chosefusion4(self, festures, targets):

        iaVal = []
        minlossIndex = []

        classAcc = []

        for i in range(len(festures)):
            fusiFx4 = self.fc3(festures[i])
            fusiFx4sig = F.sigmoid(fusiFx4)
            fusiFx_acc = F.softmax(fusiFx4, dim=1)
            acclist = []
            for k in range(fusiFx_acc.size(0)):
                tempacc = fusiFx_acc[k].data.tolist()
                maxV = max(tempacc)
                tempacc.pop(tempacc.index(max(tempacc)))
                SedmaxV = max(tempacc)
                acclist.append((maxV, SedmaxV, fusiFx_acc[k][targets[k]].data.item()))
                pass
            classAcc.append(torch.mean(torch.tensor(acclist), dim=0))
            fusiFx_loss = F.cross_entropy(fusiFx4sig, targets)
            iaVal.append(fusiFx_loss.item())

            pass
        minlossV = min(iaVal)

        for j in range(len(iaVal)):
            if minlossV == iaVal[j]:
                minlossIndex.append(j)
                pass
            pass
        ias = []
        iasindex = []
        if len(minlossIndex) > 1:

            for inde in minlossIndex:
                templist = classAcc[inde].data.tolist()
                if templist[0] != templist[-1]:
                    ias.append(pow(natureE, templist[0]) - pow(natureE, templist[-1]))
                    pass
                else:
                    ias.append(pow(natureE, templist[1]) - pow(natureE, templist[-1]))
                    pass
                iasindex.append(inde)
                pass
            indexi = iasindex[ias.index(min(ias))]
            pass
        else:
            indexi = iaVal.index(minlossV)
            pass
        weighx = torch.zeros_like(festures[1], requires_grad=True)
        if indexi < 1:
            outfeats = weighx + festures[0]
            pass
        else:
            outfeats = festures[indexi]
            pass

        return outfeats

    def compw(self, ecginfo_x6, tar):
        ecgW = torch.ones((ecginfo_x6.size(0), 1))
        ws = torch.softmax(ecginfo_x6, dim=1)
        prelabel = torch.argmax(ws, dim=1)
        for i in range(ecginfo_x6.size(0)):
            if not (prelabel == tar)[i]:
                ecgW[i] = ws[i][tar[i]]
                pass
            pass
        return ecgW

    def forward(self, x1, x2, x3, x4, x5, x6, labels, othersParms=None):
        x1 = self.conv1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x2 = self.conv1(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)

        x3 = self.conv1(x3)
        x3 = self.relu(x3)
        x3 = self.maxpool(x3)

        x4 = self.conv1(x4)
        x4 = self.relu(x4)
        x4 = self.maxpool(x4)

        x5 = self.conv4(x5)
        x5 = self.relu(x5)
        x5 = self.maxpool(x5)

        x5 = x5.view(x5.size(0), -1)
        x5 = self.fc4(x5)

        x6 = self.conv3(x6)
        x6 = self.relu(x6)
        x6 = self.maxpool(x6)

        x6 = x6.view(x6.size(0), -1)
        x6 = self.fc5(x6)

        wx1 = 1.0
        wx2 = 1.0
        wx3 = 1.0
        wx4 = 1.0

        x = torch.cat((wx1 * x1, wx2 * x2, wx3 * x3, wx4 * x4), dim=1)
        oldx = x
        if self.fusmodel == 0:
            x = oldx
            fusx = 0
            pass
        if self.fusmodel == 1:
            ac, bd = self.flattenmodule(x1, x2, x3, x4)  #
            # cona = self.con1d(a)
            # conb = self.con1d(b)
            conac = self.con1d(ac)
            conbd = self.con1d(bd)
            fusx = self.fusionmodule(conac, conbd, oldx)  #

            fusx = self.maxpool(fusx)
            fusx = fusx.view(fusx.size(0), -1)
            fusx = self.fc2(fusx)
            pass

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        if self.flagatt == 'A':
            maxfea, meanfea = self.attention128(x, fusx)

            conbinf = torch.cat((torch.unsqueeze(maxfea, 1), torch.unsqueeze(meanfea, 1)), dim=1)
            conbinf = torch.squeeze(self.con2d(conbinf))
            weightedx = self.relu(conbinf)

            if weightedx.size(0) != x.size(0):
                weightedx = weightedx.expand_as(x)
                pass
            x = self.chosefusion4(festures=[x + fusx, weightedx, weightedx + x + fusx], targets=labels)

            pass
        # if self.flagatt == 'B':
        #     a = cosine_similarity(x, fusx)
        #
        #     maxfea, meanfea = self.attention128(x, fusx)
        #
        #     conbinf = torch.cat((torch.unsqueeze(maxfea, 1), torch.unsqueeze(meanfea, 1)), dim=1)
        #     conbinf = torch.squeeze(self.con2d(conbinf))
        #     weightedx = self.relu(conbinf)
        #     x = weightedx + x + fusx
        #     pass


        localbeta = self.beta
        # localbeta = math.tan(self.beta * math.pi / 2.0)
        x = self.chosefusion4(festures=[x, x - localbeta * x5], targets=labels)

        coeff = math.tan(self.gamma * math.pi / 2.0)
        x = x + coeff * x6
        x = self.fc3(x)

        return x, x
