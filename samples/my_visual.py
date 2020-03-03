import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import imageio
import skimage.io
import cv2
import scipy.misc 

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
MASK_DIR = os.path.join(ROOT_DIR,"my_npys")
FIG_DIR= os.path.join(ROOT_DIR,"my_figs_png")

def main():
    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    #data
    color=[32,178,170]
    fpath = os.path.join(os.path.dirname(__file__), "splits", "eigen", "{}_files.txt")
    fo=open(fpath.format("test"))
    train_filenames = fo.readlines()
    fo.close()
    for line in train_filenames:
        folder,frame_index1,side=line.strip('\n').split(' ')
        frame_index=int(frame_index1)
        image_path=get_image_path(folder,frame_index,side)
        image = skimage.io.imread(image_path)
        npy_path=get_npy_path(folder,frame_index,side)
        mask=np.load(npy_path)
        mask_rev = np.where(mask==1,0,1)
        #margin=margin_mask(mask).astype(np.uint8)
        #margin_dst=get_dilated_mask(margin)
        margin_dst=get_dilated_mask(mask_rev.astype(np.uint8))
        print("aaaaaaaaaa")
        img=apply_mask(image,margin_dst,color)
        
        fig, ax = plt.subplots(1,dpi=300)
        height, width = image.shape[:2]
        ax.set_ylim(height, 0)
        ax.set_xlim(0, width )
        ax.axis('off')
        fig.set_size_inches(width/100.0/3.0, height/100.0/3.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.margins(0,0)
        ax.imshow(img.astype(np.uint8))
        #image_to_save=np.swapaxes(margin_dst,0,1)
        dirs=os.path.join(FIG_DIR, '{}'.format(folder))
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        #np.save(os.path.join(FIG_DIR,folder,"{:010d}.{}".format(frame_index,"npy")),margin_dst)
        plt.imsave(os.path.join(FIG_DIR,folder,"{:010d}.{}".format(frame_index ,"png")),img.astype(np.uint8))
        #print(margin_dst.shape)
        #plt.show()

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 0,
                                   image[:, :, c],
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c])
    return image

def margin_mask(image):
    """Locate the margins of the mask
    """
    img_ctr = image[1:-1,1:-1]*8
    img_lu = image[0:-2, 0:-2]
    img_ru = image[0:-2, 2:]
    img_u = image[0:-2, 1:-1]
    img_l = image[1:-1, 0:-2]
    img_r = image[1:-1, 2:]
    img_ld = image[2:, 0:-2]
    img_rd = image[2:, 2:]
    img_d = image[2:, 1:-1]
    img_tmp = img_ctr-img_d-img_l-img_ld-img_r-img_rd-img_ru-img_u-img_lu
    margin = np.where(img_tmp>0,1,0)
    pad_width1 = ((1,1),(1,1))
    data_p = np.pad(margin, pad_width=pad_width1, mode='constant', constant_values=0)
    print(data_p.shape)
    return data_p

def get_dilated_mask(image):
   
    kernel = np.ones((5,5),np.uint8)

    dst = cv2.dilate(image,kernel)
    return dst


def get_image_path(folder, frame_index, side):
    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    f_str = "{:010d}.{}".format(frame_index, "jpg")
    datas=os.path.abspath("/harddisk_1/xuefeng_data/kitti_data")
    image_path = os.path.join(datas, folder, "image_0{}".format(side_map[side]),"data", f_str)
    return image_path


def get_npy_path(folder,frame_index,side):
    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    npy_path = os.path.join(ROOT_DIR, "my_npys",'{}'.format(folder),'{:010d}_{}.npy'.format(frame_index,side_map[side]))
    return npy_path

    
if __name__=="__main__":
    main()
