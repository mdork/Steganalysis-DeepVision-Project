from tqdm import tqdm
import glob
import jpegio as jpio
import numpy as np

## Write dictionary
img_path = '/export/data/mdorkenw/data/alaska2/'
folders = ['Cover', 'JUNIWARD', 'JMiPOD', 'UERD']

# for tech in folders:
for folder in folders:
    dct_coeff = []
    images = glob.glob(img_path + folder + "/*.jpg")

    for img in tqdm(images):
        jpegStruct = jpio.read(img)
        dct_coeff.append(np.stack(jpegStruct.coef_arrays, axis=2))

    assert len(images) == len(dct_coeff)
    np.save(img_path + 'DCT_coefficients_' + folder + '.npy', dct_coeff)