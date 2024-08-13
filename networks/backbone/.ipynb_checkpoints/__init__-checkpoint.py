from networks.backbone import resnet, xception, drn, mobilenet,mixstyle,ada,randconv,ffnet,ccsdg

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    elif backbone == 'mixstyle':
        return mixstyle.Mixstyle(output_stride, BatchNorm)
    elif backbone == 'ADA':
        return ada.ADA(output_stride, BatchNorm)
    elif backbone == 'RandConv':
        return randconv.RandConv(output_stride, BatchNorm)
    elif backbone == 'FFNET':
        return ffnet.FFNET(output_stride, BatchNorm)
    elif backbone == 'CCSDG':
        return ccsdg.CCSDG(output_stride, BatchNorm)
    else:
        raise NotImplementedError
