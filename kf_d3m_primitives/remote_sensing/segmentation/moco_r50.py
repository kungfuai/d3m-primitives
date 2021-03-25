import types

import torch
from torch import nn
from rsp.moco_r50.resnet import resnet50


def moco_r50(
    state_dict_path=None,
    map_location=torch.device("cpu"),
    encoder_freeze=True,
    load_weights=True,
):

    model = resnet50(in_channels=12, num_classes=128)

    if state_dict_path:
        state_dict = torch.load(state_dict_path, map_location=map_location)[
            "state_dict"
        ]

        dim_mlp = model.fc.weight.shape[1]
        model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)

        for k in list(state_dict.keys()):
            if "encoder_q" not in k:
                del state_dict[k]

        state_dict = {
            k.replace("module.", "").replace("encoder_q.", ""): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(state_dict)

    model.fc = nn.Sequential()

    def forward(self, x):
        stages = [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

        features = []
        for i in range(len(stages)):
            x = stages[i](x)
            features.append(x)

        return features

    model.forward = types.MethodType(forward, model)

    if encoder_freeze:
        for name, param in model.named_parameters():
            if "bn" in name or "downsample.1" in name:
                continue
            else:
                param.requires_grad = False

    return model