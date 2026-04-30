from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import cv2
from config import args
import numpy as np
from sklearn.model_selection import StratifiedKFold




class GetData_raw(Dataset):
    def __init__(self, path1, transform, count, is_reshape=True):
        super(GetData_raw, self).__init__()
        self.path = path1
        self.transform = transform
        self.is_reshape = is_reshape
        self.count = count  # 4
        self.dataset = []
        print(os.path.join(self.path, 'label.txt'))
        self.dataset.extend(open(os.path.join(self.path, 'label.txt')).readlines())

    def __getitem__(self, index):
        str1 = self.dataset[index].strip()
        imgdata = []
        for i in range(self.count):
            imgpath = os.path.join(self.path, str1.split(',')[i])
            im = cv2.imread(imgpath)
            if self.is_reshape == True:
                im = cv2.resize(im, (120, 120))
            imgdata.append(self.transform(im))
        label = int(str1.split(',')[-1])
        return [imgdata[i] for i in range(self.count)] + [label]

    def __len__(self):
        return len(self.dataset)



if args.sde_cde3C == 'sde4C':
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [67.44569946, 75.65099565, 75.28079819]],
                                     std=[x / 255.0 for x in [21.13942046, 19.46179206, 19.71228527]])
    normalizeecgn1 = transforms.Normalize(mean=[-66.73242274274789], std=[713.2221458449152])

    pass




class GetData_newrawloso2(Dataset):
    # separately transform to four inputs
    def __init__(self, path1, fold, transform, count, foldid, is_reshape=False):
        super(GetData_newrawloso2, self).__init__()
        self.path = path1
        if args.sde_cde3C == 'sde4C':
            self.paths = r'../dataset4C/'
            pass
        self.ecgpaths = r'../'
        self.transform = transform
        self.transformnew = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        self.transformecg = transforms.Compose([
            transforms.ToTensor(),
            normalizeecgn1,
        ])
        self.flags = foldid
        self.is_reshape = is_reshape
        self.count = count  # 4
        self.dataset = []

        # training data
        if self.flags == 2:
            self.dataset.extend(self.path)
            if args.is_augment:
                self.dataset.extend(self.path)
                pass
            pass
        #  testing data
        if self.flags == 0:

            if args.sde_cde3C == 'sde4C':
                fpath = 'label4Cn1.txt'
                pass
            f = open(os.path.join(self.path, fpath))
            imgdat = f.readlines()
            for i in range(len(imgdat)):
                if imgdat[i][0:imgdat[i].find('/')] == fold:
                    self.dataset.append(imgdat[i])
                pass
            f.close()
            pass

    def __getitem__(self, index):
        # print(f"*******index={index}")
        str1 = self.dataset[index].strip()
        imgdata = []
        # Obtain RGB information
        for i in range(self.count):
            if i < self.count - 1:
                imgpath = os.path.join(self.paths, str1.split(',')[i])
                im = cv2.imread(imgpath)
                if self.is_reshape:
                    im = cv2.resize(im, (120, 120))
                    pass
                # imgdata.append(self.transform(im))
                if len(self.dataset) > 500:
                    if index < len(self.dataset) / 2:
                        imgdata.append(self.transform(im))
                    else:
                        imgdata.append(self.transformnew(im))
                        pass
                else:
                    imgdata.append(self.transformnew(im))
                pass

            pass

        ecgpath = os.path.join(self.ecgpaths, str1.split(',')[-2])
        with open(ecgpath, 'r+', encoding='utf-8') as ecgp:
            ecgtxt = ecgp.readlines()
            handfs = np.zeros((len(ecgtxt), len(ecgtxt[0].split(','))),dtype=np.float32)
            for i in range(len(ecgtxt)):
                tempf = [float(fv) for fv in ecgtxt[i].split(',')]
                handfs[i, :] = np.asarray(tempf)
                pass
        imgdata.append(self.transformecg(handfs))
        label = int(str1.split(',')[-1])
        return [imgdata[i] for i in range(self.count)] + [label]

    def __len__(self):
        return len(self.dataset)





class GetData_split(Dataset):
    def __init__(self, path1, txt_name, transform, count, is_reshape=True):
        super(GetData_split, self).__init__()
        self.path = path1
        self.transform = transform
        self.is_reshape = is_reshape
        self.count = count  # 4
        self.dataset = []
        self.dataset.extend(open(os.path.join(self.path, txt_name)).readlines())

    def __getitem__(self, index):
        str1 = self.dataset[index].strip()
        imgdata = []
        for i in range(self.count):
            imgpath = os.path.join(self.path, str1.split(',')[i])
            im = cv2.imread(imgpath)
            if self.is_reshape == True:
                im = cv2.resize(im, (120, 120))
            imgdata.append(self.transform(im))
        label = int(str1.split(',')[-1])
        return [imgdata[i] for i in range(self.count)] + [label]

    def __len__(self):
        return len(self.dataset)
