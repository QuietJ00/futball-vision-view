"""Named pitch lines from a pre-trained network, for the experimental hybrid calibration.

The network is HRNetV2-W48 with one heatmap per line class; each class's two line ends are its
strongest heatmap peaks. This file is an independent implementation of that published
architecture so the released weights can be read as plain tensors (`weights_only=True`).

Weights: `SV_lines` from https://github.com/mguti97/PnLCalib/releases (CC BY 4.0, Gutiérrez-Pérez
& Agudo, "PnLCalib: Sports Field Registration via Points and Lines Optimization"). They are not
stored in this repository. Set AUTOHL_LINE_WEIGHTS to the file to enable this module.
"""

import os

import cv2
import numpy as np

LINE_CLASSES = {
    # network channel: (name, index in pitch.model_segments). Goal frames are not ground lines.
    0: ('left_box_near', 5), 1: ('left_box_front', 6), 2: ('left_box_far', 7),
    3: ('right_box_near', 11), 4: ('right_box_front', 12), 5: ('right_box_far', 13),
    12: ('halfway', 4), 13: ('near_touchline', 1), 14: ('left_goal_line', 0),
    15: ('right_goal_line', 3), 16: ('far_touchline', 2),
    17: ('left_goal_area_near', 8), 18: ('left_goal_area_front', 9), 19: ('left_goal_area_far', 10),
    20: ('right_goal_area_near', 14), 21: ('right_goal_area_front', 15), 22: ('right_goal_area_far', 16),
}
CHANNELS = 24  # 23 line classes and a background channel
SEGMENT_OF = {name: segment for name, segment in LINE_CLASSES.values()}
MIRROR = {'left': 'right', 'right': 'left'}


def mirrored(name):
    """The class this line would have if the network has the two ends of the pitch swapped."""
    side, _, rest = name.partition('_')
    return f'{MIRROR[side]}_{rest}' if side in MIRROR else name


def _network(outputs):
    import torch
    import torch.nn.functional as F
    from torch import nn

    def norm(channels):
        return nn.BatchNorm2d(channels, momentum=.1)

    def conv3(a, b, stride=1):
        return nn.Conv2d(a, b, 3, stride, 1, bias=False)

    class Basic(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1, self.bn1 = conv3(channels, channels), norm(channels)
            self.conv2, self.bn2 = conv3(channels, channels), norm(channels)

        def forward(self, x):
            y = F.relu(self.bn1(self.conv1(x)), inplace=True)
            return F.relu(self.bn2(self.conv2(y))+x, inplace=True)

    class Bottleneck(nn.Module):
        def __init__(self, inputs, planes):
            super().__init__()
            self.conv1, self.bn1 = nn.Conv2d(inputs, planes, 1, bias=False), norm(planes)
            self.conv2, self.bn2 = conv3(planes, planes), norm(planes)
            self.conv3, self.bn3 = nn.Conv2d(planes, planes*4, 1, bias=False), norm(planes*4)
            self.downsample = None
            if inputs != planes*4:
                self.downsample = nn.Sequential(nn.Conv2d(inputs, planes*4, 1, bias=False), norm(planes*4))

        def forward(self, x):
            y = F.relu(self.bn1(self.conv1(x)), inplace=True)
            y = F.relu(self.bn2(self.conv2(y)), inplace=True)
            y = self.bn3(self.conv3(y))
            return F.relu(y+(x if self.downsample is None else self.downsample(x)), inplace=True)

    class Module(nn.Module):
        def __init__(self, channels):
            super().__init__()
            count = len(channels)
            self.branches = nn.ModuleList(nn.Sequential(*[Basic(c) for _ in range(4)]) for c in channels)
            rows = []
            for i in range(count):
                row = []
                for j in range(count):
                    if j == i:
                        row.append(None)
                    elif j > i:
                        row.append(nn.Sequential(nn.Conv2d(channels[j], channels[i], 1, bias=False), norm(channels[i])))
                    else:
                        down = []
                        for k in range(i-j):
                            out = channels[i] if k == i-j-1 else channels[j]
                            down.append(nn.Sequential(conv3(channels[j], out, 2), norm(out), nn.ReLU(True)))
                        row.append(nn.Sequential(*down))
                rows.append(nn.ModuleList(row))
            self.fuse_layers = nn.ModuleList(rows)

        def forward(self, xs):
            xs = [branch(x) for branch, x in zip(self.branches, xs)]
            out = []
            for i, row in enumerate(self.fuse_layers):
                total = xs[i]
                for j, layer in enumerate(row):
                    if j < i:
                        total = total+layer(xs[j])
                    elif j > i:
                        total = total+F.interpolate(layer(xs[j]), size=xs[i].shape[-2:], mode='bilinear',
                                                    align_corners=False)
                out.append(F.relu(total))
            return out

    class HRNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1, self.bn1 = conv3(3, 64, 2), norm(64)
            self.conv2, self.bn2 = conv3(64, 64, 2), norm(64)
            self.layer1 = nn.Sequential(Bottleneck(64, 64), *[Bottleneck(256, 64) for _ in range(3)])
            self.transition1 = nn.ModuleList([
                nn.Sequential(conv3(256, 48), norm(48), nn.ReLU(True)),
                nn.Sequential(nn.Sequential(conv3(256, 96, 2), norm(96), nn.ReLU(True)))])
            self.stage2 = nn.Sequential(Module([48, 96]))
            self.transition2 = nn.ModuleList([
                None, None, nn.Sequential(nn.Sequential(conv3(96, 192, 2), norm(192), nn.ReLU(True)))])
            self.stage3 = nn.ModuleList(Module([48, 96, 192]) for _ in range(4))
            self.transition3 = nn.ModuleList([
                None, None, None, nn.Sequential(nn.Sequential(conv3(192, 384, 2), norm(384), nn.ReLU(True)))])
            self.stage4 = nn.ModuleList(Module([48, 96, 192, 384]) for _ in range(3))
            self.head = nn.Sequential(nn.Sequential(
                nn.Conv2d(784, 784, 1), norm(784), nn.ReLU(True), nn.Conv2d(784, outputs, 1), nn.Sigmoid()))

        def forward(self, x):
            x = self.conv1(x)
            skip = x
            x = F.relu(self.bn2(self.conv2(F.relu(self.bn1(x), inplace=False))), inplace=True)
            x = self.layer1(x)
            xs = [layer(x) for layer in self.transition1]
            xs = self.stage2[0](xs)
            xs = [xs[0], xs[1], self.transition2[2](xs[-1])]
            for module in self.stage3:
                xs = module(xs)
            xs = [xs[0], xs[1], xs[2], self.transition3[3](xs[-1])]
            for module in self.stage4:
                xs = module(xs)
            size = xs[0].shape[-2:]
            x = torch.cat([xs[0]]+[F.interpolate(y, size=size, mode='bilinear', align_corners=False)
                                   for y in xs[1:]], 1)
            return self.head(torch.cat([F.interpolate(x, scale_factor=2, mode='nearest'), skip], 1))

    return HRNet()


class SemanticLines:
    def __init__(self, weights=None, device=None):
        import torch
        self.torch = torch
        weights = weights or os.environ.get('AUTOHL_LINE_WEIGHTS')
        if not weights or not os.path.exists(weights):
            raise FileNotFoundError('AUTOHL_LINE_WEIGHTS 에 SV_lines 가중치 경로를 지정해 주세요.')
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.net = _network(CHANNELS)
        # weights_only reads tensors alone, so the file cannot run code when loaded.
        self.net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        self.net.to(self.device).eval()

    def heatmaps(self, image):
        small = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)
        tensor = self.torch.from_numpy(small[..., ::-1].copy()).float().permute(2, 0, 1)[None]/255
        with self.torch.no_grad():
            return self.net(tensor.to(self.device))[0].cpu().numpy()

    def detect(self, image, frame=0, threshold=.3):
        """Line ends per class in full-size pixels, in the structure the geometry code expects."""
        maps = self.heatmaps(image)
        lines = []
        for channel, (name, _) in LINE_CLASSES.items():
            ends = _two_peaks(maps[channel], threshold)
            if ends is not None:
                points, confidence = ends
                lines.append({'class_id': name, 'points': (points*image.shape[1]/maps.shape[2]).tolist(),
                              'confidence': confidence})
        return {'frame': frame, 'lines': lines,
                'quality': {'visible_classes': len(lines), 'coverage': float((maps[:-1].max(0) > threshold).mean())}}


def _two_peaks(heatmap, threshold, separation=10):
    peak = cv2.dilate(heatmap, np.ones((2*separation+1,)*2, np.uint8))
    y, x = np.nonzero((heatmap == peak) & (heatmap >= threshold))
    if len(x) < 2:
        return None
    order = np.argsort(-heatmap[y, x])[:2]
    return np.c_[x[order], y[order]].astype(float), float(heatmap[y[order], x[order]].min())
