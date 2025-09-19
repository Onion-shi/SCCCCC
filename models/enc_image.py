import torch.nn as nn, timm

class ConvNeXtTinyEncoder(nn.Module):
    """
    输出四级特征：[(B,96,H/4,W/4), (B,192,H/8,W/8), (B,384,H/16,W/16), (B,768,H/32,W/32)]
    """
    def __init__(self, pretrained=True, out_indices=(0,1,2,3)):
        super().__init__()
        self.backbone = timm.create_model("convnext_tiny", pretrained=pretrained, features_only=True, out_indices=out_indices)
        self.feat_channels = self.backbone.feature_info.channels()

    def forward(self, x):
        return self.backbone(x)
