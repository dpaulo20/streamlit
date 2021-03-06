

""" script python qui contient toutes les fonctions utiles de transformation """

from glob import glob
from PIL import Image
import numpy as np
import cv2
from imgaug.augmentables.segmaps import SegmentationMapOnImage
from imgaug.imgaug import draw_text
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy


def np_resize(img, input_shape):
    """
    Reshape a numpy array, which is input_shape=(height, width), 
    as opposed to input_shape=(width, height) for cv2
    """
    height, width = input_shape
    return cv2.resize(img, (width, height))


def mask_to_rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_to_mask(rle, input_shape):
    '''
    Convertit une chaine RLE(run length encoding) en un tableau numpy

    Valeurs en entrée: 
    rle (str): string of rle encoded mask
    input_shape (int, int): dimension du mask

    Valeur retournées: 
    numpy.array: tableau numpy pour le masque
    '''
    width, height = input_shape[:2]
    
    mask= np.zeros(width*height).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    return mask.reshape(height, width).T


def build_masks(rles, input_shape, reshape=None):
    depth = len(rles)
    if reshape is None:
        masks = np.zeros((*input_shape, depth))
    else:
        masks = np.zeros((*reshape, depth))
    
    for i, rle in enumerate(rles):
        if type(rle) is str:
            if reshape is None:
                masks[:, :, i] = rle_to_mask(rle, input_shape)
            else:
                mask = rle_to_mask(rle, input_shape)
                reshaped_mask = np_resize(mask, reshape)
                masks[:, :, i] = reshaped_mask
    
    return masks


def build_rles(masks, reshape=None):
    width, height, depth = masks.shape
    
    rles = []
    
    for i in range(depth):
        mask = masks[:, :, i]
        
        if reshape:
            mask = mask.astype(np.float32)
            mask = np_resize(mask, reshape).astype(np.int64)
        
        rle = mask_to_rle(mask)
        rles.append(rle)
        
    return rles

######## LOSS FUNCTIONS ##################

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

########################################

def get_labels(dataset, image_id):
    ''' Function to get the labels for the image by name'''
    im_df = dataset[dataset['ImageId'] == image_id]
    im_df = im_df[im_df['EncodedPixels'] != '-1'].groupby('Label').count()
    
    index = im_df.index
    all_labels = ['Fish', 'Flower', 'Gravel', 'Sugar']
    
    labels = ''
    
    for label in all_labels:
        if label in index:
            labels = labels + ' ' + label
    
    return labels


def get_mask_by_image_id(dataset, image_path, label):
    '''
    Function to visualize several segmentation maps.
    INPUT:
        image_id - filename of the image
    RETURNS:
        np_mask - numpy segmentation map
    '''
    im_df = dataset[dataset['ImageId'] == image_path.split('/')[-1]]

    image = np.asarray(Image.open(image_path))

    rle = im_df[im_df['Label'] == label]['EncodedPixels'].values[0]
    if rle != '-1':
        dim = (image.shape[0], image.shape[1])
        np_mask = rle_to_mask(rle, dim)
        np_mask = np.clip(np_mask, 0, 1)
    else:
        np_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
    return np_mask


def create_segmap(dataset, image_path):
    '''
    Helper function to create a segmentation map for an image by image filename
    '''
    # open the image
    image = np.asarray(Image.open(image_path))
    
    # get masks for different classes
    fish_mask = get_mask_by_image_id(dataset, image_path, 'Fish')
    flower_mask = get_mask_by_image_id(dataset, image_path, 'Flower')
    gravel_mask = get_mask_by_image_id(dataset, image_path, 'Gravel')
    sugar_mask = get_mask_by_image_id(dataset, image_path, 'Sugar')
    
    # label numpy map with 4 classes
    segmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
    segmap = np.where(fish_mask == 1, 1, segmap)
    segmap = np.where(flower_mask == 1, 2, segmap)
    segmap = np.where(gravel_mask == 1, 3, segmap)
    segmap = np.where(sugar_mask == 1, 4, segmap)
    
    # create a segmantation map
    segmap = SegmentationMapOnImage(segmap, shape=image.shape, nb_classes=5)
    
    return segmap


def draw_labels(image, np_mask, label):
    '''
    Function to add labels to the image.
    '''
    if np.sum(np_mask) > 0:
        x,y = 0,0
        x,y = np.argwhere(np_mask==1)[0]
                
        image = draw_text(image, x, y, label, color=(255, 255, 255), size=50)
    return image


def draw_segmentation_maps(dataset, image_path):
    '''
    Helper function to draw segmentation maps and text.
    '''
    # open the image
    image = np.asarray(Image.open(image_path))
    
    # get masks for different classes
    fish_mask = get_mask_by_image_id(dataset, image_path, 'Fish')
    flower_mask = get_mask_by_image_id(dataset, image_path, 'Flower')
    gravel_mask = get_mask_by_image_id(dataset, image_path, 'Gravel')
    sugar_mask = get_mask_by_image_id(dataset, image_path, 'Sugar')
    
    # label numpy map with 4 classes
    segmap = create_segmap(dataset, image_path)
    
    # draw the map on image
    image = np.asarray(segmap.draw_on_image(image)).reshape(image.shape)
    
    image = draw_labels(image, fish_mask, 'Fish')
    image = draw_labels(image, flower_mask, 'Flower')
    image = draw_labels(image, gravel_mask, 'Gravel')
    image = draw_labels(image, sugar_mask, 'Sugar')
    
    return image


def get_masks(dataset, image_id, image_size, nb_classes):
    im_df = dataset[dataset['ImageId'] == image_id]
    data = np.ndarray( shape = (image_size, image_size, nb_classes))
    all_mask = np.zeros((image_size, image_size, nb_classes), dtype=np.uint8)
    
    for label in (im_df['CategoryId']):
          for index, lab in enumerate(label):
                rle = im_df['EncodedPixels'].values[0][index]
                dim = (1400, 2100)
                np_mask = rle_to_mask(rle, dim)
                np_mask = np.clip(np_mask, 0, 1)
                np_mask_reshape = cv2.resize(np_mask,(image_size, image_size), interpolation = cv2.INTER_AREA)
                for col in range(image_size):
                        for row in range(image_size):
                            all_mask[row,col,lab]=np_mask_reshape[row,col]
    return all_mask


def apply_mask(image, mask, color, alpha=0.5):
    ''' Applique un mask simple à l'image '''

    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color*255,
                                  image[:, :, c])
    return image


##Creation d'un DATAGENERATOR renvoyant les images et masks en 512*512


def cloud_data_gen(df, lists, batch_size, directory_image):
    while True:
        ix = np.random.choice(np.arange(len(lists)), batch_size)
        imgs = []
        masks = []
        for i in ix:
            # images
            filnames=df['ImageId'][i]
            img=cv2.imread(directory_image+filnames)
            img=cv2.resize(img,(512, 512), interpolation = cv2.INTER_AREA)           
        #    array_img = img_to_array(resized_img) / 255
            imgs.append(img)
            # masks
            mask=get_masks(df, filnames, 512, 4)
            masks.append(mask)
        imgs = np.array(imgs)
        masks = np.array(masks)
        yield imgs, masks


def main():
    pass

if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()

