import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network


CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


#CLASSES = ('__background__','person','bike','motorbike','car','bus')


def vis_detections(im, class_name, dets,ax, thresh=0.5):
    """This function draws all the detected bounding boxes."""

    #returns the index value at which dets > threshold value 
    #considers only the last column as it has values as [[x,y,h,w,thresh]]
    inds = np.where(dets[:, -1] >= thresh)[0]
    
    #if there is nothing found in the inds then it returns
    if len(inds) == 0:
        return

    #this part of the code shows all bounding boxes for all locations where the value is above thresh
    #the bounding box is made and the text is annotated along with the class scores
    
    #traverse through the indexes in inds where dets >= thresh
    for i in inds:
        #traverse through all indexes in dets and use the first 4 values [x,y,h,w]
        bbox = dets[i, :4]
        #use the score from the last value in dets called thresh
        score = dets[i, -1]

        #matplotlib.pyplot patch.Rectangle function to draw bounding boxes
        #here parameters include ((x,y), width , height , fill , edgecolor , linewidth)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )


        #text annotation code foraddding text to the picture to show class and its score
        #blue bounding box is added here around the text
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    #add a title to the axis , different from ax.suptitle which addes title to figure , not worth mentioning
    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)

    #configure the pyplot for plotting the image
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    #im_file = os.path.join('/home/corgi/Lab/label/pos_frame/ACCV/training/000001/',image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()

    #im_detect() function detects the  classe of an object in an image given object proposals.
    """
    the function im_detect() has the following definition:
    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    #the code below for image indexing is probably to change rbg to bgr or vice versa
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    #setting confidence-threshold and non-maximum-supression threshold
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3

    #iterate over all classes and find the boxes and scores for each individual boxes
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background

        #boxes (ndarray): R x (4*K) array of predicted bounding boxes
        #use the boxes value for a specific class
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        #use a specific score for a particular class
        cls_scores = scores[:, cls_ind]

        #stacks arrays in sequence horizontally
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        
        #nms() function returns the indexes of those boxes that we should keep by considering the IOU values
        keep = nms(dets, NMS_THRESH)

        #keep only the required boxes and their scores , others are discarded
        dets = dets[keep, :]

        #send the final decided bounding boxes to the above defined vis_detections() function
        vis_detections(im, cls, dets, ax, thresh=CONF_THRESH)

def parse_args():
    """
    Parse input arguments. Helps with the command line argument input.
    """
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    #we need to mention the pretrained path else it will raise an error
    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))

    # init session
    #allow_soft_placement automatically chooses an existing and supported device to run the operations in case the specified one doesn't exist
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    # load model
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.restore(sess, args.model)
   
    #sess.run(tf.initialize_all_variables())

    print '\n\nLoaded network {:s}'.format(args.model)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(sess, net, im)

    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']


    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(sess, net, im_name)

    #finally show all the bounding boxes and class scores along with the image
    plt.show()