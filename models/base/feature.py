r""" Extracts intermediate features from given backbone network & layer ids """


def extract_feat_vgg(img, backbone, feat_ids, bottleneck_ids=None, lids=None):
    r"""Extract intermediate features from VGG"""
    feats = []
    feat = img
    for lid, module in enumerate(backbone.features):
        feat = module(feat)
        if lid in feat_ids:
            feats.append(feat.clone())
    return feats


def extract_feat_res(img, backbone, feat_ids, bottleneck_ids, lids):
    r"""Extract intermediate features from ResNet"""
    feats = []

    # Layer 0
    feat = backbone.conv1.forward(img)
    feat = backbone.bn1.forward(feat)
    feat = backbone.relu.forward(feat)
    feat = backbone.maxpool.forward(feat)

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = backbone.__getattr__("layer%d" % lid)[bid].conv1.forward(feat)
        feat = backbone.__getattr__("layer%d" % lid)[bid].bn1.forward(feat)
        feat = backbone.__getattr__("layer%d" % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__("layer%d" % lid)[bid].conv2.forward(feat)
        feat = backbone.__getattr__("layer%d" % lid)[bid].bn2.forward(feat)
        feat = backbone.__getattr__("layer%d" % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__("layer%d" % lid)[bid].conv3.forward(feat)
        feat = backbone.__getattr__("layer%d" % lid)[bid].bn3.forward(feat)

        if bid == 0:
            res = backbone.__getattr__("layer%d" % lid)[bid].downsample.forward(res)

        feat += res

        if hid + 1 in feat_ids:
            feats.append(feat.clone())

        feat = backbone.__getattr__("layer%d" % lid)[bid].relu.forward(feat)

    return feats


def extract_feat_clip(x, model, feat_ids, bottleneck_ids, lids):
    feats = []

    def stem(x):
        for conv, bn in [
            (model.visual.conv1, model.visual.bn1),
            (model.visual.conv2, model.visual.bn2),
            (model.visual.conv3, model.visual.bn3),
        ]:
            x = model.visual.relu(bn(conv(x)))
        x = model.visual.avgpool(x)
        return x

    feat = x.type(model.visual.conv1.weight.dtype)
    feat = stem(feat)

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = model.visual.__getattr__("layer%d" % lid)[bid].conv1.forward(feat)
        feat = model.visual.__getattr__("layer%d" % lid)[bid].bn1.forward(feat)
        feat = model.visual.__getattr__("layer%d" % lid)[bid].relu.forward(feat)
        feat = model.visual.__getattr__("layer%d" % lid)[bid].conv2.forward(feat)
        feat = model.visual.__getattr__("layer%d" % lid)[bid].bn2.forward(feat)
        feat = model.visual.__getattr__("layer%d" % lid)[bid].relu.forward(feat)
        feat = model.visual.__getattr__("layer%d" % lid)[bid].avgpool.forward(feat)
        feat = model.visual.__getattr__("layer%d" % lid)[bid].conv3.forward(feat)
        feat = model.visual.__getattr__("layer%d" % lid)[bid].bn3.forward(feat)

        if bid == 0:
            res = model.visual.__getattr__("layer%d" % lid)[bid].downsample.forward(res)

        feat += res

        if hid + 1 in feat_ids:
            feats.append(feat.clone())

        feat = model.visual.__getattr__("layer%d" % lid)[bid].relu.forward(feat)

    return feats
