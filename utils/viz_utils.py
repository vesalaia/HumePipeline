"""
Visualization
"""
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
try:
    from detectron2 import model_zoo  
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.utils.visualizer import ColorMode
except:
    pass

import copy
from utils.cv_utils import convert_from_image_to_cv2, convert_from_cv2_to_image

def onImage(imagePath, outfile, model):
    image = cv2.imread(imagePath)
    print("onImage: imagePath", imagePath)
    print("onImage: outfile", outfile)
    outputs = model(image)

    v = Visualizer(image[:, :, ::-1], metadata={}, instance_mode=ColorMode.SEGMENTATION)
    pic = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(20, 15))
    plt.imshow(pic.get_image()[:, :, ::-1])
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    plt.savefig(outfile)
    plt.show()


def plotBBoxes(imgpath, boxes):
    x = cv2.imread(imgpath)
    color = (255, 0, 0)
    thickness = 4
    image = copy.deepcopy(x)
    for bb in boxes:
        start_point = bb[0]
        end_point = bb[1]
        image = cv2.rectangle(image, start_point, end_point, color, thickness)

    plt.imshow(image)
    plt.show()
def drawLinePolygon(file_path, coords):
    img = cv2.imread(file_path)
    height = img.shape[0]
    width = img.shape[1]
    x1, y1 = coords.min(axis=0)
    x2, y2 = coords.max(axis=0)

    mask = np.zeros((height, width), dtype=np.uint8)
    # points = np.array([[[10,150],[150,100],[300,150],[350,100],[310,20],[35,10]]])
    cv2.fillPoly(mask, [coords], (255))
    #    img = convert_from_image_to_cv2(img)
    res = cv2.bitwise_and(img, img, mask=mask)
    res = convert_from_cv2_to_image(res)
    cropped_image = res.crop((x1, y1, x2, y2))

    plt.imshow(cropped_image)
    plt.show()


def formatScore(x):
    score = " {:.2f}".format(x * 100)
    return score
def plotPage(dataset, idx, outfile=None):
    fig = plt.figure(figsize=(18, 18))
    alpha = 1 
    beta = 0.6 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    # train on the GPU or on the CPU, if a GPU is not available
    img, target = dataset.__getitem__(idx)
    filename = dataset.__getfullname__(idx)
    image = np.array(img)
    masks = target['masks']
    boxes = target['boxes']
    labels = target['labels']
    for j in range(len(target['labels'])):

        label = get_key(labels[j].item())
        plt.imshow(img)
 
        red_map = np.zeros_like(masks[j]).astype(np.uint8)
        green_map = np.zeros_like(masks[j]).astype(np.uint8)
        blue_map = np.zeros_like(masks[j]).astype(np.uint8)
        # apply a randon color mask to each object
        color = COLORS[random.randrange(0, len(COLORS))]
        red_map[masks[j] == 1], green_map[masks[j] == 1], blue_map[masks[j] == 1]  = color
        # combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        #convert the original PIL image into NumPy format

        # convert from RGN to OpenCV BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # apply mask on the image
        cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
        # draw the bounding boxes around the objects
        start_point = (int(boxes[j][0].item()),int(boxes[j][1].item()))
        end_point = (int(boxes[j][2].item()),int(boxes[j][3].item()))
        cv2.rectangle(image, start_point, end_point, color=color, 
                      thickness=2)
            
        # put the label text above the objects
        cv2.putText(image , get_key(labels[j].item()), start_point, 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
                    thickness=2, lineType=cv2.LINE_AA)
    if outfile != None:
        if not os.path.exists(os.path.dirname(outfile)):
            os.makedirs(os.path.dirname(outfile))
        plt.savefig(outfile)
    else:
        plt.imshow(image)   

