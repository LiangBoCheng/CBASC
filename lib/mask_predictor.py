import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def US2(x):
    """if size!=None:
        return F.interpolate(x, size=size, mode='bilinear')
    else:"""
    return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
def US4(x):
    """if size!=None:
        return F.interpolate(x, size=size, mode='bilinear')
    else:"""
    return F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
def US8(x):
    """if size!=None:
        return F.interpolate(x, size=size, mode='bilinear')
    else:"""
    return F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)

class GSCC(nn.Module):
    def __init__(self, in_planes):
        super().__init__()

        # -------------------------
        # Spatial Attention Pooling
        # -------------------------
        self.visual_attn = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // 4, 1, 1, bias=False)
        )

        # -------------------------
        # Language Attention Pooling
        # -------------------------
        self.lang_attn = nn.Linear(in_planes, 1)

        # -------------------------
        # Shared Semantic Projection
        # -------------------------
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // 2, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, l_v, v_l):
        """
        l_v : [B,C,H,W]   language-aware visual feature
        v_l : [B,N,C]     vision-aware language feature
        """

        B, C, H, W = l_v.shape

        # ====================================================
        # Global Visual Representation (Spatial Attention Pooling)
        # ====================================================

        # [B,1,H,W]
        score = self.visual_attn(l_v)

        # Spatial Softmax
        score = score.view(B, 1, -1)
        attn_v = F.softmax(score, dim=-1)
        attn_v = attn_v.view(B, 1, H, W)

        # Weighted Sum
        v_g = (l_v * attn_v).sum(dim=(2, 3), keepdim=True)

        # ====================================================
        # Global Language Representation (Attention Pooling)
        # ====================================================

        attn_l = self.lang_attn(v_l)          # [B,N,1]
        attn_l = F.softmax(attn_l, dim=1)

        l_g = (attn_l * v_l).sum(dim=1)       # [B,C]
        l_g = l_g.unsqueeze(-1).unsqueeze(-1)

        # ====================================================
        # Shared Semantic Projection
        # ====================================================

        z_v = self.mlp(v_g)

        z_l = self.mlp(l_g)

        # ====================================================
        # Global Semantic Consistency
        # ====================================================

        s = z_v * z_l

        # ====================================================
        # Residual Calibration
        # ====================================================

        weight = self.sigmoid(s)

        out = l_v + weight * l_v

        return out

class SimpleDecoding(nn.Module):
    def __init__(self, c4_dims, factor=2):
        super(SimpleDecoding, self).__init__()

        c4_size = c4_dims
        c3_size = c4_dims//(factor**1)
        c2_size = c4_dims//(factor**2)
        c1_size = c4_dims//(factor**3)

        self.gscc1 = GSCC(c1_size)
        self.gscc2 = GSCC(c2_size)
        self.gscc3 = GSCC(c3_size)
        self.gscc4 = GSCC(c4_size)

        self.conv_cat1 = nn.Sequential(BasicConv2d(c4_size + c3_size, c3_size, 3, padding=1),
                                       BasicConv2d(c3_size, c3_size, 3, padding=1))

        self.conv_cat2 = nn.Sequential(BasicConv2d(c3_size + c2_size, c2_size, 3, padding=1),
                                       BasicConv2d(c2_size, c2_size, 3, padding=1))

        self.conv_cat3 = nn.Sequential(BasicConv2d(c2_size + c1_size, c1_size, 3, padding=1),
                                       BasicConv2d(c1_size, c1_size, 3, padding=1))

        self.conv1_1 = nn.Conv2d(c1_size, 2, 1)
        self.linear1 = nn.Linear(768, c1_size)
        self.linear2 = nn.Linear(768, c2_size)
        self.linear3 = nn.Linear(768, c3_size)
        self.linear4 = nn.Linear(768, c4_size)

    def forward(self, x_c4, x_c3, x_c2, x_c1, x_l4, x_l3, x_l2, x_l1, l_mask):
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)
        x_l4 = x_l4 * l_mask
        x_l3 = x_l3 * l_mask
        x_l2 = x_l2 * l_mask
        x_l1 = x_l1 * l_mask

        l4 = self.linear4(x_l4.permute(0, 2, 1))
        l3 = self.linear3(x_l3.permute(0, 2, 1))
        l2 = self.linear2(x_l2.permute(0, 2, 1))
        l1 = self.linear1(x_l1.permute(0, 2, 1))

        x_c4 = self.gscc4(x_c4, l4)
        x_c3 = self.gscc3(x_c3, l3)
        x_c2 = self.gscc2(x_c2, l2)
        x_c1 = self.gscc1(x_c1, l1)

        x_c4 = US2(x_c4)
        x_c3 = self.conv_cat1(torch.cat((x_c4, x_c3), 1))

        x_c3 = US2(x_c3)
        x_c2 = self.conv_cat2(torch.cat((x_c3, x_c2), 1))

        x_c2 = US2(x_c2)
        x_c1 = self.conv_cat3(torch.cat((x_c2, x_c1), 1))

        x = self.conv1_1(x_c1)

        return x