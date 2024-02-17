
from natsort import natsorted
import glob
import logging
import cv2
from tqdm import tqdm
import numpy as np
import math
import argparse
import sys
import shutil

from auxilary.utils import *
from auxilary.simplex import Simplex_CLASS as simplex


def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='none', help='Path to the Config file.')
    return parser.parse_args()

# function for flipping image
def flip_lr(img, prob):
    if prob <= 0.5:
        return img, 'F'
    else:
        return np.fliplr(img), 'T'

def flip_ud(img, prob):
    if prob <= 0.5:
        return img, 'F'
    else:
        return np.flipud(img), 'T'

# function for rotating image
def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)
    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST ## GROUND TRUTH는INTER_NEAREST로 해야함.
    )

    return result

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    scaleFactor = 0.5
    #scaleFactor = 0.8

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * scaleFactor)
    x2 = int(image_center[0] + width * scaleFactor)
    y1 = int(image_center[1] - height * scaleFactor)
    y2 = int(image_center[1] + height * scaleFactor)

    return image[y1:y2, x1:x2]

def rotate_crop_random(img, angle):
    image_height, image_width = img.shape[0:2]

    image_rotated = rotate_image(img, angle)
    image_rotated_cropped = crop_around_center(image_rotated,
                                               *largest_rotated_rect(image_width, image_height, math.radians(angle)))

    return image_rotated_cropped, str(angle)

# resize Image
def resize_image(image, new_dimensions):
    return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_CUBIC)


def magnify_image(image, magnify_random):
    # Apply magnification using cv2.resize with the magnification factor
    magnified_image = cv2.resize(image, None, fx=magnify_random, fy=magnify_random, interpolation=cv2.INTER_CUBIC)
    # Determine the crop coordinates to bring the image back to the original size
    crop_x = (magnified_image.shape[1] - image.shape[1]) // 2
    crop_y = (magnified_image.shape[0] - image.shape[0]) // 2

    # Crop the magnified image to the original size
    final_image = magnified_image[crop_y:crop_y + image.shape[0], crop_x:crop_x + image.shape[1]]
    return final_image


# function for gamma correction
def adjust_gamma(img, gamma):
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # Apply the gamma correction using the lookup table
    return cv2.LUT(img, table)

# Function to distort image
def elastic_transform(image, alpha=100, sigma=10, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
     and https://github.com/rwightman/tensorflow-litterbox/blob/ddeeb3a6c7de64e5391050ffbb5948feca65ad3c/litterbox/fabric/image_processing_common.py#L220
    """
    if random_state < 0.5:
        return image

    shape_size = image.shape[:2]

    # Downscaling the random grid and then upsizing post filter
    # improves performance. Approx 3x for scale of 4, diminishing returns after.
    grid_scale = 4
    alpha //= grid_scale  # Does scaling these make sense? seems to provide
    sigma //= grid_scale  # more similar end result when scaling grid used.
    grid_shape = (shape_size[0]//grid_scale, shape_size[1]//grid_scale)

    blur_size = int(4 * sigma) | 1
    rand_x = cv2.GaussianBlur(
        (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
        ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    rand_y = cv2.GaussianBlur(
        (random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
        ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    if grid_scale > 1:
        rand_x = cv2.resize(rand_x, shape_size[::-1])
        rand_y = cv2.resize(rand_y, shape_size[::-1])

    grid_x, grid_y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
    grid_x = (grid_x + rand_x).astype(np.float32)
    grid_y = (grid_y + rand_y).astype(np.float32)

    distorted_img = cv2.remap(image, grid_x, grid_y,
        borderMode=cv2.BORDER_REFLECT_101, interpolation=cv2.INTER_LINEAR)

    return distorted_img


# Function to add noise to image
def noisy_image(img, alpha, random_state=None):
    if random_state < 0.5:
        return img

    # Generate noise
    simplexObj = simplex()
    img_size = (img.shape[0], img.shape[1])
    noise = simplexObj.rand_2d_octaves(img_size, 6, 0.6)
    # Convert image to float [0, 1]
    image_array = img.astype(np.float32) / 255
    # Normalize Noise to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())  
    # Blend noise with original image
    image_array = (1 - alpha) * image_array + alpha * noise[..., np.newaxis]
    # Convert back to uint8 [0, 255]
    image_array = (image_array * 255).astype(np.uint8)
    return image_array


def performAugmentation(configPath, mode = 'train'):

    config = readConfig(configPath)
    log_dir = config["log"]
    f = open(log_dir + mode+"_augmentationLog.txt", "w")
    # GET CONFIGURATION DETAILS
    augment_num_per_img = config["augmentPerImage"]
    tile_width = config["finalTileWidth"]
    tile_height = config["finalTileHeight"]

    # Read images and corresponding labels
    modeList = ['orig_train', 'orig_validation']
    if mode == 'train':
        slidingDir = config["out_dir"]+modeList[0]+"/"
    else:
        slidingDir = config["out_dir"]+modeList[1]+"/"

    labeled_imgs = natsorted(glob.glob(f'{slidingDir}labels/*'))
    raw_imgs = natsorted(glob.glob(f'{slidingDir}images/*'))

    # assert len of images and labels are equal else raise error
    assert len(labeled_imgs) == len(raw_imgs), "Number of images and labels are not equal"

    logging.basicConfig(level=logging.DEBUG)

    base_dir = config["augmented_dir"]+mode+"/"
    createDir([config["augmented_dir"], base_dir])


    # print stats
    logging.debug("Number of images: " + str(len(labeled_imgs)))
    logging.debug("Number of labels: " + str(len(raw_imgs)))
    logging.debug("Number of augmentations per image: " + str(augment_num_per_img))
    logging.debug("Tile width: " + str(tile_width))
    logging.debug("Tile height: " + str(tile_height))
    logging.debug("Sliding direction: " + str(slidingDir))
    logging.debug("Augmented directory: " + str(base_dir))

    # run once flag
    run_once = True


    count = 0

    for i in tqdm(range(len(labeled_imgs))):
        # read image and label
        raw_image = cv2.imread(raw_imgs[i], cv2.IMREAD_COLOR)
        label = cv2.imread(labeled_imgs[i], cv2.IMREAD_GRAYSCALE)


        if run_once:
            logging.debug("raw image size:" + str(raw_image.shape))
            logging.debug("label size:" + str(label.shape))
            run_once = False

        # Set random probabilities
        flip_random = np.random.uniform(size=augment_num_per_img)  # [0,1]
        rotate_random = np.random.randint(low=0, high=360, size=augment_num_per_img)  # [0, 360]
        magnify_random = np.random.uniform(low=1, high=1.5, size=augment_num_per_img)  # [1, 1.5]
        elastic_random = np.random.uniform(size=augment_num_per_img)  # [0, 1]
        noise_random = np.random.uniform(size=augment_num_per_img)  # [0, 1]


        for j in range(augment_num_per_img//3):
            # 3 times because 3 images are generated per iteration

            # flip image
            flipped_image, flip_flag = flip_lr(raw_image, flip_random[j])
            flipped_label, flip_flag = flip_lr(label, flip_random[j])

            flipped_image, flip_flag = flip_ud(flipped_image, flip_random[j])
            flipped_label, flip_flag = flip_ud(flipped_label, flip_random[j])

            # rotate randomly without blank
            modImage, flag = rotate_crop_random(flipped_image, rotate_random[j])
            modLabel, flag = rotate_crop_random(flipped_label, rotate_random[j])

            # magnify randomly
            modImage = magnify_image(modImage, magnify_random[j])
            modLabel = magnify_image(modLabel, magnify_random[j])

            # add noise 
            modImage = noisy_image(modImage, alpha=0.3, random_state=noise_random[j])


            # elastic transform
            if elastic_random[j] <= 0.5:
                modImage = elastic_transform(modImage, alpha=300, sigma=30, random_state=elastic_random[j])
                modLabel = elastic_transform(modLabel, alpha=300, sigma=30, random_state=elastic_random[j])


            # resize image and label
            modImage = resize_image(modImage, (tile_width, tile_height))
            modLabel = resize_image(modLabel, (tile_width, tile_height))

            # gamma adjustment
            gamma_values = [ 1, 1.5]

            for gamma_value in gamma_values:
                corrected_image = adjust_gamma(modImage, gamma_value)
                new_img_name = str(count) +".png"
                new_label_name = str(count) + "_label.png"
                cv2.imwrite(base_dir + new_img_name, corrected_image)
                cv2.imwrite(base_dir + new_label_name , modLabel)
                f.write(new_img_name+"\n"+new_label_name+"\n")

                count += 1

            # save image and label
            new_img_name = str(count) +".png"
            new_label_name = str(count) + "_label.png"
            cv2.imwrite(base_dir + new_img_name, modImage)
            cv2.imwrite(base_dir + new_label_name, modLabel)
            f.write(new_img_name+"\n"+new_label_name+"\n")
        
            count += 1

    f.close()

def organizeTestImages(configPath):
    config = readConfig(configPath)
    in_dir = config["out_dir"]+"orig_test/"
    out_dir = config["augmented_dir"]+"test/"

    createDir([out_dir])

    for i in tqdm(range(len(glob.glob(in_dir + "images/*")))):
        shutil.copy(in_dir + "images/" + str(i) + ".png", out_dir + str(i) + ".png")
        shutil.copy(in_dir + "labels/" + str(i) + ".png", out_dir + str(i) + "_label.png")

if __name__ == '__main__':
    
    # Read Config
    args = arg_init()
    configPath = args.config

    if configPath == 'none':
        print('Please specify the path to the config file using --config <path to config.sys>')
        sys.exit()

    print("Performing Augmentation on Train Images")
    performAugmentation(configPath , 'train')
    print("Performing Augmentation on Val Images")
    performAugmentation(configPath, 'val')
    print("Organizaing Test Images")
    organizeTestImages(configPath)

    

