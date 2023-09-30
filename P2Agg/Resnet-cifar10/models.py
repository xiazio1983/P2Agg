import torch
from torchvision import models
import torch.nn as nn


def get_model(name="vgg16", pretrained=True):

    if name == "resnet18":

        model = models.resnet18(pretrained=pretrained)
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.MaxPool2d(1 ,1, 0)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)


    elif name == "resnet50":
        # 使用在ImageNet数据集上预训练的ResNet-50作为源模型。指定pretrained=True自动下载预训练的模型参数
        model = models.resnet50(pretrained=pretrained)
    elif name == "densenet121":
        model = models.densenet121(pretrained=pretrained)
    elif name == "alexnet":
        model = models.alexnet(pretrained=pretrained)
    elif name == "vgg16":
        model = models.vgg16(pretrained=pretrained)
    elif name == "vgg19":
        model = models.vgg19(pretrained=pretrained)
    elif name == "inception_v3":
        model = models.inception_v3(pretrained=pretrained)
    elif name == "googlenet":
        model = models.googlenet(pretrained=pretrained)

    if torch.cuda.is_available():
        return model.cuda()
    else:
        return model