import os
import pandas as pd
import numpy as np
import cv2
import imgaug as ia
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from sklearn.utils import shuffle

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNEL = 3
CLASSES = ["Background", "Person"]
N_CLASSES = 2


class DataGenerator:
    def __init__(self, 
                 data_dir, 
                 anno_paths, 
                 augment_func,
                 img_height=IMG_HEIGHT,
                 img_width=IMG_WIDTH,
                 img_channel=IMG_CHANNEL,
                 batch_size=36,
                 n_classes=N_CLASSES,
                 augmentation=True,
                 task="train"):
        
        self.data_dir = data_dir
        self.anno_paths = anno_paths
        self.augment_func = augment_func
        self.batch_size = batch_size
        self.task = task
        self.current_index = 0
        # self.current_test = 0
        self.img_height = img_height
        self.img_width = img_width
        self.img_channel = img_channel
        self.n_classes = n_classes
        self.augmentation = augmentation
        self.image_paths, self.label_paths = self.load_image_paths("input", "target")
        
    def load_image_paths(self, input_dir, target_dir):
        image_paths = []
        label_paths = []
        
        for anno_path in self.anno_paths:
            anno_dir = os.path.dirname(anno_path)
            df_file_names = pd.read_csv(anno_path, header=None)
            file_names = df_file_names[0].values
            
            for fn in file_names:
                if "DS_Store" in fn: continue
                img_train_dir = os.path.join(anno_dir, input_dir)
                img_label_dir = os.path.join(anno_dir, target_dir)
                img_train_path = os.path.join(img_train_dir, fn)
                img_label_path = os.path.join(img_label_dir, fn)
        
                image_paths.append(img_train_path)
                label_paths.append(img_label_path)
            
        image_paths = np.array(image_paths)
        label_paths = np.array(label_paths)

        return image_paths, label_paths
    
    def get_n_examples(self):
        return len(self.image_paths)
        
    def load_image(self, img_path):
        image = cv2.imread(img_path, 1)
        image = image[..., ::-1]
        # print(len(np.unique(image[..., 0])))
        return image

    def load_images(self, image_paths):
        images = []
        preprocessors = [self.resize_img, self.mean_substraction]
        
        for img_path in image_paths:
            image = self.load_image(img_path)
            image = self.preprocessing(image, preprocessors=preprocessors)
            images.append(image)
        
        images = np.array(images)
        return images
    
    def load_labels(self, label_paths):
        labels = []
        
        for lbl_path in label_paths:
            label = self.load_image(lbl_path)
            label = self.resize_img(label, interpolation=cv2.INTER_NEAREST)
            label = label / 255
            label[label < 0.5] = 0
            label[label >= 0.5] = 1
            label = self.parse_label(label)
            labels.append(label)
        
        labels = np.array(labels)
        
        return labels
    
    def parse_label(self, seg_image):
        label = seg_image
        label = label[:, :, 0]
        seg = np.zeros((self.img_height, self.img_width, self.n_classes))
        
        for i in range(self.n_classes):
            seg[:, :, i] = (label == i).astype(int)
        
        seg = np.reshape(seg, (-1, self.n_classes))
        return seg
    
    def load_batch_pair_in_pair(self, img_paths, label_paths):
        images = []
        segs = []
        preprocessors = [self.resize_img, self.mean_substraction]
        
        for i in range(len(img_paths)):
            image = self.load_image(img_paths[i])
            # image = self.preprocessing(image, preprocessors)
            
            seg = self.load_image(label_paths[i])
            # seg = self.preprocessing(seg, [self.resize_img])
            # print(seg.dtype)
            # seg = seg.astype(np.uint8)
            # seg = self.parse_label(seg)
            # seg = np.argmax(seg, axis=-1).astype(np.int32)
            segmap = SegmentationMapsOnImage(seg[:, :, 0], shape=image.shape, nb_classes=np.max(seg[:, :, 0]) + 1)
            
            if self.augmentation:
                image, segmap = self.augment_func(image=image, segmentation_maps=segmap)
            
            seg[:, :, 0] = segmap.get_arr_int()
            image = self.preprocessing(image, preprocessors)
            
#             seg = self.preprocessing(seg, [self.resize_img])
            seg = self.resize_img(seg, interpolation=cv2.INTER_NEAREST)
            seg = seg / 255
            seg[seg > 0.5] = 1
            seg[seg <= 0.5] = 0
            seg = self.parse_label(seg)
            
            images.append(image)
            segs.append(seg)
        
        return np.array(images), np.array(segs)
    
    def show_sample_image(self):
        idx = np.random.choice(len(self.image_paths), 1)[0]
        img_path = self.image_paths[idx]
        seg_path = self.label_paths[idx]
        
        img = self.load_image(img_path)
        seg = self.load_image(seg_path)
        # segmap = SegmentationMapsOnImage(seg[..., 0], shape=img.shape)
        segmap = SegmentationMapsOnImage(seg, shape=img.shape)
        
        img_aug, seg_aug = self.augment_func(image=img, segmentation_maps=segmap)
        
        ia.imshow(np.hstack([
            img[..., ::-1],
            seg
        ]))
        
        # seg[..., 0] = seg_aug.get_arr_int()
        seg = seg_aug.get_arr_int()
        # print(np.nonzero(seg[..., 0]))

        # non_index = np.nonzero(seg[..., 0])
        # for i in range(len(non_index[0])):
        #     print(seg[..., 0][non_index[0][i], non_index[1][i]])

        ia.imshow(np.hstack([
            img_aug[..., ::-1],
            seg
        ]))
        
        seg = seg / 255
        seg[seg < 0.5] = 0
        seg[seg >= 0.5] = 1
        
        ia.imshow(np.hstack([
            img_aug[..., ::-1],
            seg
        ]))
    
    def resize_img(self, image, interpolation=cv2.INTER_AREA):
        # h, w = image.shape[:2]
        # print(h, w)
        # p_h, p_w = 0, 0
        #
        # if h>w:
        #     image = imutils.resize(image, width=self.img_width)
        #     p_h = int((image.shape[0] - self.img_height) / 2)
        # else:
        #     image = imutils.resize(image, height=self.img_height)
        #     p_w = int((image.shape[1] - self.img_width) / 2)
        #
        # image = image[p_w:image.shape[1]-p_w, p_h: image.shape[0]-p_h]
        image = cv2.resize(image, (self.img_height, self.img_width), interpolation=interpolation)
        return image
    
    def mean_substraction(self, image, mean=None, image_val=0.017):
        if mean is None:
            mean = [103.94, 116.78, 123.68]
        image = image.astype("float32")
        image[:, :, 0] -= mean[0]
        image[:, :, 1] -= mean[1]
        image[:, :, 2] -= mean[2]
        image *= image_val
        image = image[..., ::-1]
        
        return image
    
    def preprocessing(self, image, preprocessors):
        for p in preprocessors:
            image = p(image)
        
        return image
    
    def load_batch(self):
        if self.current_index + self.batch_size >= len(self.image_paths):
            self.current_index = 0
            self.image_paths, self.label_paths = shuffle(self.image_paths, self.label_paths, random_state=42)
            
        img_batch_paths = self.image_paths[self.current_index:self.current_index+self.batch_size]
        seg_batch_paths = self.label_paths[self.current_index:self.current_index+self.batch_size]
        self.current_index += self.batch_size
        inputs, segs = self.load_batch_pair_in_pair(img_batch_paths, seg_batch_paths)
        
        return inputs, segs
    
    def generate(self):
        while True:
            images, labels = self.load_batch()
            
            yield (images, labels)
