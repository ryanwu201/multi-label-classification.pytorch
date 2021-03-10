import numpy as np
import cv2


class CAMGenerator:
    def __init__(self, finalconv_name, model):
        self.finalconv_name = finalconv_name
        self.model = model
        # switch to evaluate mode
        self.model.eval()

        # hook the feature extractor
        self.features_blobs = []
        me = self

        def hook_feature(module, input, output):
            me.features_blobs.append(output.data.cpu().numpy())

        self.model._modules.get(finalconv_name).register_forward_hook(hook_feature)

        # get the softmax weight
        params = list(self.model.parameters())
        self.weight_softmax = np.squeeze(params[-2].cpu().data.numpy())

    def generate(self, feature_conv_index, class_idx):
        feature_conv = self.features_blobs[feature_conv_index]
        # generate the class activation maps upsample to 256x256
        size_upsample = (256, 256)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = self.weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam
