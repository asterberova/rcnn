"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 nucleus.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
import cv2
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
DSLAB_DATA = '/data/s2732815/rcnn/'
# COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_WEIGHTS_PATH = os.path.join(DSLAB_DATA, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
# DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_LOGS_DIR = os.path.join(DSLAB_DATA, "logs")

# Results directory
# Save submission files here
# RESULTS_DIR = os.path.join(ROOT_DIR, "results/nucleus/")
RESULTS_DIR = os.path.join(DSLAB_DATA, "results/cells/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.

############################################################
#  Configurations
############################################################

class CellsConfig(Config):
    """Configuration for training on the cell segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "cells"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 6

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + cells

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 628 // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, 20 // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between cell and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "none"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 0.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class CellsInferenceConfig(CellsConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    # pad64/crop/square/none
    IMAGE_RESIZE_MODE = "none"
    IMAGE_MIN_SCALE = 0.0

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    fig.tight_layout()
    return ax


############################################################
#  Dataset
############################################################

class CellsDataset(utils.Dataset):

    def load_cells(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset cells, and the class cells
        self.add_class("cells", 1, "cells")

        # Which subset?
        # "val": use hard-coded list above
        # "train", "val", "test": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        # assert subset in ["train", "val", "test"]
        subset_dir = subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)

        # Get image ids from directory names
        image_ids = next(os.walk(dataset_dir))[1]
        # if subset == "train":
        #     image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

        # Add images
        for image_id in image_ids:
            self.add_image(
                "cells",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id, "images/{}.png".format(image_id)))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

        # Read mask files from .png image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cells":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir, subset, n_epochs=50):
    """Train the model."""
    # Training dataset.
    dataset_train = CellsDataset()
    dataset_train.load_cells(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CellsDataset()
    dataset_val.load_cells(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    # print("Train network heads")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=20,
    #             augmentation=augmentation,
    #             layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=n_epochs,
                augmentation=augmentation,
                layers='all')


############################################################
#  RLE Encoding
############################################################

def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset, mask_score, count_statistics):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = CellsDataset()
    dataset.load_cells(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    APs = list()
    F1_scores = list()
    precisions_dict = {}
    recall_dict = {}
    num_of_confident_masks = 0
    for image_id in dataset.image_ids:
        print(f"Detection on image: {image_id}")
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)

        save_data = os.path.join(submit_dir, dataset.image_info[image_id]["id"])
        if not os.path.exists(save_data):
            os.makedirs(save_data)
        save_data = os.path.join(submit_dir, dataset.image_info[image_id]["id"], 'masks')
        if not os.path.exists(save_data):
            os.makedirs(save_data)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}/predictions.png".format(submit_dir, dataset.image_info[image_id]["id"]))

        id_mask = 0
        N = len(r['scores'])
        for i in range(N):
            # Score
            score = r['scores'][i]
            # Mask
            mask = r['masks'][:, :, i]
            # print(f'Mask shape {mask.shape}')
            # if score > 0.8:
            if score > mask_score:
                mask_img = mask * 255
                # print(mask_img)
                cv2.imwrite('{}/{}/masks/{}.png'.format(submit_dir, dataset.image_info[image_id]["id"], str(id_mask)), mask_img)
                id_mask += 1
                num_of_confident_masks += 1

        print(f"count_statistics: {count_statistics}, type: {type(count_statistics)}")
        if count_statistics:
            # load image, bounding boxes and masks for the image id and graound truth
            image2, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
                dataset, config, image_id)
            print("Original image shape: ",
                  modellib.parse_image_meta(image_meta[np.newaxis, ...])["original_image_shape"][0])

            # Run object detection
            # results = model.detect_molded(np.expand_dims(image, 0), np.expand_dims(image_meta, 0), verbose=1)
            # Display results
            # r = results[0]
            # New prediction https://github.com/matterport/Mask_RCNN/issues/2165
            # scaled_image = modellib.mold_image(image, config)
            # sample = np.expand_dims(scaled_image, 0)
            # yhat = model.detect(sample, verbose=0)
            # r = yhat[0]

            try:
                visualize.display_differences(
                    image,
                    gt_bbox, gt_class_id, gt_mask,
                    r['rois'], r['class_ids'], r['scores'], r['masks'],
                    dataset.class_names, ax=get_ax(),
                    show_box=False, show_mask=False,
                    iou_threshold=0.5, score_threshold=0.5)
                plt.savefig("{}/{}/difference.png".format(submit_dir, dataset.image_info[image_id]["id"]))
            except Exception as e:
                print(f"Visualize display_differences failed on {str(e)}")

            # calculate statistics, including AP
            AP, precisions, recalls, _ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"],
                                                    r['masks'])
            precisions_dict[image_id] = np.mean(precisions)
            recall_dict[image_id] = np.mean(recalls)
            # store
            if AP != 'nan':
                APs.append(AP)
            f1 = (2 * (np.mean(precisions) * np.mean(recalls))) / (np.mean(precisions) + np.mean(recalls))
            if f1 != 'nan':
                F1_scores.append(f1)
            # print(f"AP: {AP}, TYPE AP: {type(AP)}")
            # print(f"F1: {f1}, TYPE F1: {type(f1)}")


    if count_statistics:
        # calculate the mean AP and mean F1 across all tested images
        mAP = np.mean(APs)
        mF1 = np.mean(F1_scores)

        # # Save mAP to txt file
        file_path = os.path.join(submit_dir, "statistics.txt")
        with open(file_path, "w") as f:
            f.write(f'Mean AP: {str(mAP)} \n')
            f.write(f'Mean F1: {str(mF1)} \n')
            f.write(f'------------------------------------------\n')
            f.write(f'APs: {str(APs)} \n')
            f.write(f'F1 scores: {str(F1_scores)} \n')
        # Save precision and recall
        file_path = os.path.join(submit_dir, "precision.txt")
        with open(file_path, 'w') as file:
            file.write(json.dumps(str(precisions_dict)))  # use `json.loads` to do the reverse
        file_path = os.path.join(submit_dir, "recall.txt")
        with open(file_path, 'w') as file:
            file.write(json.dumps(str(recall_dict)))  # use `json.loads` to do the reverse

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)
    print(f"Predicted in total {num_of_confident_masks} masks with score > {mask_score}")


############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for cell counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    parser.add_argument('--mask_score', required=False, type=float,
                        metavar="Mask score to detect and save mask",
                        help="Threshold of mask score to be detected")
    parser.add_argument('--stats', required=False, type=int, default=False,
                        metavar="Compute statistics of detection",
                        help="0/1 : don't compute/compute")
    parser.add_argument('--epoch', required=False, type=int,
                        metavar="Compute statistics of detection",
                        help="Should compute statistics of detection")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CellsConfig()
    else:
        config = CellsInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset, args.epoch)
    elif args.command == "detect":
        print(f"DETECTION, stats: {args.stats}")
        detect(model, args.dataset, args.subset, args.mask_score, args.stats)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
