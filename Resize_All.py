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
                if fname[-4:] == '.npy':
                    continue
                if remote:
                    if root[-2] != 'y':
                        save_fname = root[-16:].replace('/', '_') + '_' + fname[-11:-4] + '.npy'
                    else:
                        save_fname = root[-15:].replace('/', '_') + '_' + fname[-11:-4] + '.npy'
                else:
                    if root[-2] != 'y':
                        save_fname = root[-16:].replace('/', '_') + '_' + fname[-11:-4] + '.npy'
                    else:
                        save_fname = root[-15:].replace('\\', '_') + '_' + fname[-11:-4] + '.npy'
                
                fullpath = os.path.join(out_path, save_fname)
                
                if not os.path.exists(fullpath):
                    full_im = plt.imread(os.path.join(root, fname))
                    small_im = skimage.transform.resize(full_im, (x,y), order=1,
                                                        mode='edge', clip=True, preserve_range=True,
                                                        anti_aliasing=True, anti_aliasing_sigma=True)
                    
                    np.save(fullpath, small_im)
                    
                    

if __name__ == '__main__':
    print('Running Resize_All.py', file=open('Resize_512x512.out', 'a'))
    
    remote = True
    
    if remote:
        root_path = "/groups/CS156b/data/train"
        out_path = "/groups/CS156b/2023/BbbBbbB/Train_512x512"
    else:
        root_path = "D:\\cs156"
        out_path = "D:\\cs156\\images_200x200"
    
    pid_min = None # None
    
    print('Resizing training set!', file=open('Resize_512x512.out', 'a'))
    
    resize_images(root_path, out_path, x=512, y=512, remote=remote, pid_min=pid_min)
    
    print('Finished resizing training set!\n', file=open('Resize_512x512.out', 'a'))
    
    if remote:
        root_path = "/groups/CS156b/data/test"
        out_path = "/groups/CS156b/2023/BbbBbbB/Test_512x512"
    
    print('Resizing test set!', file=open('Resize_512x512.out', 'a'))
        
    resize_images(root_path, out_path, x=512, y=512, remote=remote, pid_min=pid_min)
    
    print('Finished resizing test set!\n', file=open('Resize_512x512.out', 'a'))
    
    print('Finished running Resize_All.py', file=open('Resize_512x512.out', 'a'))

    os._exit(0)