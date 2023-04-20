import os
import numpy as np
import matplotlib.pyplot as plt
import skimage


for root, subdirs, files in os.walk('/groups/CS156b/data/train'):
    if len(root) < 10:
        continue
    elif len(files) > 0:
        for fname in files:
            full_im = plt.imread(os.path.join(root, fname))
            small_im = skimage.transform.resize(full_im, (200,200), order=1,
                                                mode='edge', clip=True, preserve_range=True,
                                                anti_aliasing=True, anti_aliasing_sigma=True)
            np.save(os.path.join('/groups/CS156b/2023/BbbBbbB/images_200x200',
                                     fname[:-4], '.npy'), small_im)