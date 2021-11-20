import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

from skimage import io
from skimage.transform import resize
import utils

from dataloader import FaceData
import resnet
import cv2

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def main():
    top1 = utils.AvgrageMeter()

    train_transform, valid_transform = utils._data_transforms()
    public_valid_set =  FaceData(split='public_valid',transform=valid_transform)
    public_valid_queue = torch.utils.data.DataLoader(dataset=public_valid_set,
                                                     batch_size=128,
                                                     shuffle=False, pin_memory=True, num_workers=16)

    model = resnet.resnet50(pretrained=True, num_classes=7)
    checkpoint = utils.load('./models')
    model.load_state_dict(checkpoint[1])
    model.cuda()


    # exit()
    with torch.no_grad():
        model.eval()
        for step, (input, target) in enumerate(public_valid_queue):
                bs, ncrops, c, h, w = np.shape(input)
                input = input.cuda()
                target = target.cuda()
                input = input.view(-1, c, h, w)

                input, target = Variable(input), Variable(target)
                logits = model(input)
                logits_avg = logits.view(bs, ncrops, -1).mean(1)  # avg over crops

                prec1, prec5 = utils.accuracy(logits_avg, target, topk=(1, 5))
                n = input.size(0)
                top1.update(prec1.data, n)

        print('valid acc is %e',top1.avg)

        src = cv2.imread('./3.png', 0)
        src_o = Image.open('./3.png').convert('RGB')
        img = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, (48,48), interpolation = cv2.INTER_AREA)
        img = Image.fromarray(img)
        input = valid_transform(img)
        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        print(img.size)
        ncrops, c, h, w = np.shape(input)

        input = input.view(-1, c, h, w)
        input = input.cuda()
        input = Variable(input, volatile=True)
        out = model(input)
        out_avg = out.view(ncrops, -1).mean(0)  # avg over crops

        score = F.softmax(out_avg)
        _, predicted = torch.max(score.data, 0)

    plt.subplot(121)
    plt.imshow(src_o)
    plt.title('original image')
    plt.axis('off')

    plt.subplot(122)
    ind = 0.1 + 0.6 * np.arange(len(class_names))
    width = 0.4
    color_list = ['black', 'red', 'green', 'blue', 'cyan', 'orangered','royalblue']
    for i in range(len(class_names)):
        plt.bar(ind[i], score.data.cpu().numpy()[i], width, color=color_list[i])
    plt.title('Prediction')
    plt.xlabel('Categories')
    plt.ylabel('Predicted Score',)
    plt.xticks(ind, class_names, rotation=45, fontsize=14)
    # plt.savefig('./face_happy.png')
    plt.show()

if __name__=='__main__':
    main()