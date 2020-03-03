import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt


# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

IMAGE_DIR = os.path.join("harddisk_1","xuefeng_data","kitti_dataset")


def main():

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Directory of images to run detection on
   
    config = InferenceConfig()
    config.display()
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}


    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']
    #data
    
    fpath = os.path.join(os.path.dirname(__file__), "splits", "eigen_zhou", "{}_files.txt")
    fo=open(fpath.format("train"))
    train_filenames = fo.readlines()
    fo.close()
    results=[]
    for line in train_filenames:
        folder,frame_index1,side=line.strip('\n').split(' ')
        frame_index=int(frame_index1)
        image_path=get_image_path(folder,frame_index,side)
        image = skimage.io.imread(image_path)
    # Run detection
        # Visualize results 
        r = model.detect([image], verbose=1)[0]#把这里的image->images
        #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    #class_names, r['scores'])
        mask_to_log=np.ones([image.shape[0],image.shape[1]],dtype=np.uint8)
        zero=np.zeros([image.shape[0],image.shape[1]],dtype=np.uint8)
        N = r['rois'].shape[0]
        result=[]
        for i in range(N):
            if r['class_ids'][i] in [0,1,2,3,4,6,7,8]:
                result.append(r['class_ids'][i])
                result.append(r['scores'][i])
                mask_to_log=np.where(r['masks'][:,:,i],zero,mask_to_log)
        results.append(result)
        dirs=os.path.join(ROOT_DIR, "my_npys",'{}'.format(folder))
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        savefile = os.path.join(ROOT_DIR, "my_npys",'{}'.format(folder),'{:010d}_{}.npy'.format(frame_index,side_map[side]))
        
        np.save(savefile,mask_to_log)
    np.save(os.path.join(ROOT_DIR, "my_npys",'classify_results.npy'),results)
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def get_image_path(folder, frame_index, side):
    side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
    f_str = "{:010d}.{}".format(frame_index, "jpg")
    print(side)
    datas=os.path.abspath("/harddisk_1/xuefeng_data/kitti_data")
    image_path = os.path.join(datas, folder, "image_0{}".format(side_map[side]),"data", f_str)
    return image_path


if __name__=="__main__":
    main()



