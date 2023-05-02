import os
import numpy as np
import matplotlib.pyplot as plt
import skimage


def resize_images(root_path, out_path, x=200, y=200, remote=True, pid_min=None):
    for root, subdirs, files in os.walk(root_path):
        if len(root) < 10:
            continue
        elif len(files) > 0:
            for fname in files:
                print(root, root[29:34])
                if pid_min != None and int(root[29:34]) < pid_min:
                    continue
                if fname[-4:] == '.npy':
                    continue
                full_im = plt.imread(os.path.join(root, fname))
                small_im = skimage.transform.resize(full_im, (x,y), order=1,
                                                    mode='edge', clip=True, preserve_range=True,
                                                    anti_aliasing=True, anti_aliasing_sigma=True)
                if remote:
                    save_fname = 
                    save_fname = root[-15:].replace('/', '_') + '_' + fname[-11:-4] + '.npy'
                else:
                    save_fname = root[-15:].replace('\\', '_') + '_' + fname[-11:-4] + '.npy'
                np.save(os.path.join(out_path, save_fname), small_im)

                
if __name__ == '__main__':
    
    remote = True
    if remote:
        root_path = "/groups/CS156b/data/test"
        out_path = "/groups/CS156b/2023/BbbBbbB/test_images_200x200"
    else:
        root_path = "D:\\cs156"
        out_path = "D:\\cs156\\images_200x200"
    
    pid_min = None # None
    
    resize_images(root_path, out_path, x=200, y=200, remote=remote, pid_min=pid_min)