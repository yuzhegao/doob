import torch
import torchvision.models
import hiddenlayer as hl

from net import DoobNet

model = DoobNet()
transforms = [
    # Fold Conv, BN, RELU layers into one
    hl.transforms.Fold("Conv > BatchNorm > Relu", "ConvBnRelu"),
    hl.transforms.Fold("Conv > BatchNorm", "ConvBn"),
]

# Display graph using the transforms above
graph = hl.build_graph(model, torch.zeros([1, 3, 224, 224]), transforms=transforms)
graph.save('network.pdf')
