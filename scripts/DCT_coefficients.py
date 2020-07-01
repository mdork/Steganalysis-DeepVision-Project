from tqdm import tqdm
import glob
import jpegio as jpio
from PIL import Image
import numpy as np
from scipy.fftpack import dct
from numpy import r_

## Write dictionary
img_path = '/export/data/mdorkenw/data/alaska2/'
folders = ['Cover', 'JUNIWARD', 'JMiPOD', 'UERD']
save_path = img_path + 'DCT/'


def dct2(a):
    return dct(dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


# for tech in folders:
for folder in folders:
    images = glob.glob(img_path + folder + "/*.jpg")
    print(images)
    exit()

    for idx, img in tqdm(enumerate(images)):
        # jpegStruct = jpio.read(img)
        im = np.array(Image.open(img))
        # dct_coeff = np.stack(jpegStruct.coef_arrays, axis=2)
        # breakpoint()

        imsize = im.shape
        dct_out = np.zeros(imsize)

        # Do 8x8 DCT on image (in-place)
        for i in r_[:imsize[0]:8]:
            for j in r_[:imsize[1]:8]:
                dct_out[i:(i + 8), j:(j + 8)] = dct2(im[i:(i + 8), j:(j + 8)])

        np.save(save_path + folder + '/' + str(idx).zfill(4) + '_block.npy', dct_out)
