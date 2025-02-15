import math
import os
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from torchvision import transforms
import videotransforms
import numpy as np
import torch.nn.functional as F

from pytorch_i3d import InceptionI3d
try:
    from aslcitizen_dataset import ASLCitizen as Dataset
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Code/I3D'))
    from aslcitizen_dataset import ASLCitizen as Dataset
import cv2

from tqdm import tqdm
from operator import add

def eval_metrics(sortedArgs, label):
    res, = np.where(sortedArgs == label)
    dcg = 1 / math.log2(res[0] + 1 + 1)
    mrr = 1 / (res[0] + 1)
    if res < 1:
        return res[0], [dcg, 1, 1, 1, 1, mrr]
    elif res < 5:
        return res[0], [dcg, 0, 1, 1, 1, mrr]
    elif res < 10:
        return res[0], [dcg, 0, 0, 1, 1, mrr]
    elif res < 20:
        return res[0], [dcg, 0, 0, 0, 1, mrr]
    else:
        return res[0], [dcg, 0, 0, 0, 0, mrr]

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                       videotransforms.RandomHorizontalFlip()])
test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

video_base_path = r"C:\Users\ADITYA\OneDrive\Desktop\IIT\Code\Dataset\videos"
train_file = r"C:\Users\ADITYA\OneDrive\Desktop\IIT\Code\Dataset\splits\train.csv"
test_file = r"C:\Users\ADITYA\OneDrive\Desktop\IIT\Code\Dataset\splits\test.csv"

tag = 'experiment1d'
dataset_name = "training_subset"

train_ds = Dataset(datadir=video_base_path, transforms=train_transforms, video_file=train_file)
print(len(train_ds.gloss_dict))

test_ds = Dataset(datadir=video_base_path, transforms=test_transforms, video_file=test_file, gloss_dict=train_ds.gloss_dict)
print(len(test_ds.gloss_dict))

test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

glosses = train_ds.gloss_dict

wlasl_glosses = {}
with open(r'C:\Users\ADITYA\OneDrive\Desktop\IIT\utils\wlasl_class_list.txt') as f:
    for line in f:
        line = line.strip()
        words = line.split('\t')
        wlasl_glosses[words[1]] = int(words[0])

wlasl_subset = []
with open(r'C:\Users\ADITYA\OneDrive\Desktop\IIT\utils\wlasl_subset.txt') as f:
    for line in f:
        line = line.strip()
        wlasl_subset.append(wlasl_glosses[line])

asl_subset = []
missing_glosses = []
with open(r'C:\Users\ADITYA\OneDrive\Desktop\IIT\utils\aslcitizen_subset.txt') as f:
    for line in f:
        line = line.strip()
        if line in glosses:
            asl_subset.append(glosses[line])
        else:
            missing_glosses.append(line)
    if missing_glosses:
        print(f"Warning: Missing glosses in dictionary: {', '.join(missing_glosses)}")

asl_to_wlasl = {}
with open(r'C:\Users\ADITYA\OneDrive\Desktop\IIT\utils\aslcitizen_wlasl_subset_mapping.txt') as f:
    for line in f:
        line = line.strip()
        words = line.split('\t')
        asl_to_wlasl[words[0]] = words[1]

asl_subset.sort()
wlasl_subset.sort()

asl_gloss_map = {}
count = 0
for g in asl_subset:
    asl_gloss_map[g] = count
    count += 1

i3d = InceptionI3d(400, in_channels=3)
i3d.replace_logits(2731)
i3d.load_state_dict(torch.load(r'C:\Users\ADITYA\OneDrive\Desktop\IIT\saved_weights_jan_1a\_jan_1a75_0.736444.pt'))
i3d.cuda()

count_total = 0
count_correct = [0, 0, 0, 0, 0, 0]
conf_matrix = np.zeros((len(glosses), len(glosses)))
gloss_count = np.zeros(len(glosses))
user_stats = {}
user_counts = {}

i3d.train(False)
for data in tqdm(test_loader):
    inputs, name, labels = data
    inputs = inputs.cuda()
    t = inputs.size(2)
    labels = labels.cuda()
    users = name['user']

    gt = torch.max(labels, dim=2)[0]
    g = name['gloss'][0].strip()

    if g in asl_to_wlasl:
        g_ind = glosses[g]

        per_frame_logits = i3d(inputs)
        per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

        predictions = torch.max(per_frame_logits, dim=2)[0]
        predictions = predictions.squeeze()
        pred_subset = predictions[asl_subset]
        y_pred_tag = torch.softmax(pred_subset, dim=0)
        pred_args = torch.argsort(y_pred_tag, dim=0, descending=True)

        true_args = asl_gloss_map[g_ind]
        pred = pred_args.cpu().numpy()
        gti = true_args

        res, counts = eval_metrics(pred, gti)
        count_correct = list(map(add, counts, count_correct))
        count_total = count_total + 1

        u = users[0]
        if u not in user_counts:
            user_counts[u] = 1
            user_stats[u] = counts
        else:
            user_counts[u] = user_counts[u] + 1
            user_stats[u] = list(map(add, counts, user_stats[u]))

with open('output ' + tag + '.txt', 'w') as f:
    f.write("Total files in eval = " + str(count_total) + '\n')
    f.write("Discounted Cumulative Gain is " + str(count_correct[0]/count_total)+ '\n')
    f.write("Mean Reciprocal Rank is " + str(count_correct[5]/count_total)+ '\n')
    f.write("Top-1 accuracy is " + str(count_correct[1]/count_total)+ '\n')
    f.write("Top-5 accuracy is " + str(count_correct[2]/count_total)+ '\n')
    f.write("Top-10 accuracy is " + str(count_correct[3]/count_total)+ '\n')
    f.write("Top-20 accuracy is " + str(count_correct[4]/count_total)+ '\n')
    f.write('\n')

with open('user_stats ' + tag + '.txt', 'w') as f:
   for u in user_counts:
       f.write("User: " + u + '\n')
       f.write("Files: " + str(user_counts[u]) + '\n')
       f.write("Discounted Cumulative Gain is " + str(user_stats[u][0]/user_counts[u])+ '\n')
       f.write("Mean Reciprocal Rank is " + str(user_stats[u][5]/user_counts[u])+ '\n')
       f.write("Top-1 accuracy is " + str(user_stats[u][1]/user_counts[u])+ '\n')
       f.write("Top-5 accuracy is " + str(user_stats[u][2]/user_counts[u])+ '\n')
       f.write("Top-10 accuracy is " + str(user_stats[u][3]/user_counts[u])+ '\n')
       f.write("Top-20 accuracy is " + str(user_stats[u][4]/user_counts[u])+ '\n')
       f.write('\n')
