from keras.applications import resnet50

#model = resnet50.ResNet50()
#model.summary()
import torch.nn as nn

from torchvision import models

model_ft = models.resnet50(pretrained=True)
#model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
print(model_ft)