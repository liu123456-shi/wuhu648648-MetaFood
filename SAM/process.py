import numpy as np
from PIL import Image


def load_data(imname, maskname):
    img = Image.open(imname)
    mask = np.load(maskname)
    return img, mask

class postprocess(object):
    """Compute avarage saturation of masked area."""
    def __init__(self, **kwargs) -> None:
        self.area_ratio = kwargs.get('ratio', 0.3333)
        self.hsv_thresh = kwargs.get('hsv', 128)
        self.rgb_thresh = kwargs.get('rgb', 50)
        self.brightness = kwargs.get('brightness', 100)


    def run(self, x, mask):
        """
        Parameters
        ----------
        X : Original PIL.Image file in RGB mode.
        mask : Same shape like x while forground and backgournd are represented as 1/0.
        Returns
        -------
        A value shows if the img is over saturated.
        """
        assert x.mode == "RGB", "the input image should be in RGB mode."
        mat = np.array(x.convert('HSV'))
        h, w, _ = mat.shape
        hh, ww = mask.shape
        assert h == hh and w == ww, "the input image and mask should be the same shape."
        pixs = mat[mask > 0]
        num, edge = np.histogram(pixs[:, 1], bins=256)
        edge = edge[:-1]
        num_over_th = num[edge > self.hsv_thresh]
        pixs_over_th = np.sum(num_over_th)
        ratio = pixs_over_th / pixs.shape[0]

        return ratio

    def run2(self, x, mask):
        """
        Parameters
        ----------
        X : Original PIL.Image file in RGB mode.
        mask : Same shape like x while forground and backgournd are represented as 1/0.
        Returns
        -------
        A value shows if the img is over saturated.
        """
        assert x.mode == "RGB", "the input image should be in RGB mode."
        mat = np.array(x)
        h, w, _ = mat.shape
        hh, ww = mask.shape
        assert h == hh and w == ww, "the input image and mask should be the same shape."
        pixs = mat[mask > 0]
        mm = np.max(pixs, axis=1)
        nn = np.min(pixs, axis=1)
        mm[mm < self.brightness] = 0
        nn[mm < self.brightness] = 0
        diff = np.abs(mm - nn)
        ratio = np.nonzero(diff > self.rgb_thresh)[0].shape[0] / diff.shape[0]

        return ratio

# if __name__ == "__main__":
#     import os
#     path = 'D:\Lab\colorhouse\code'
#     files = os.listdir(path)
#     for imname in files:
#         if imname.endswith('.jpg'):
#                 exname = imname.replace('.jpg', '.npy')
#                 pilimg, mask = load_data(imname, exname)
#                 p = postprocess()
#                 # ro = p.run(pilimg, mask)
#                 ro = p.run2(pilimg, mask)
#                 print(imname , ro)