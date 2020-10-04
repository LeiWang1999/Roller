import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.feature = nn.Sequential(
            # 3, 224, 224
            nn.Conv2d(3, 48, kernel_size=11, stride=4,
                      padding=2),  # 224 - 11 + 4 / 4 + 1
            nn.ReLU(inplace=True),
            # 48, 55, 55
            nn.MaxPool2d(3, 2),
            # 48, 27, 27
            # output[128, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # output[128, 13, 13]
            nn.MaxPool2d(kernel_size=3, stride=2),
            # output[192, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[192, 13, 13]
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[128, 13, 13]
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # output[128, 6, 6]
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classfier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )
        if init_weights == True:
            self._init_weights()

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classfier(x)
        return x

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(
                    layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                nn.init.constant_(layer.bias, 0)


if __name__ == '__main__':
    inputs = torch.rand((32, 3, 224, 224))
    model = AlexNet(num_classes=5)
    outputs = model(inputs)
    print(outputs[0])
