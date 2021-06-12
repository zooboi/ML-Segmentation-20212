from imgaug import augmenters as iaa


class DataAugmentation:
    def __init__(self):
        self.IMAGE_AUGMENTATION_SEQUENCE = None
        self.IMAGE_AUGMENTATION_NUM_TRIES = 10
        self.loaded_augmentation_name = ""
    
        self.augmentation_functions = {
            "aug_all": self.load_augmentation_aug_all,
            "aug_geometric": self.load_augmentation_aug_geometric,
            "aug_non_geometric": self.load_augmentation_aug_non_geometric,
        }

    @staticmethod
    def load_augmentation_aug_geometric():
        return iaa.Sequential([
            iaa.Sometimes(0.5, iaa.Fliplr()),
            iaa.Sometimes(0.5, iaa.Rotate((-45, 45))),
            iaa.Sometimes(0.5, iaa.Affine(
                scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
                order=[0, 1],
                mode='constant',
                cval=(0, 255),
            )),
            iaa.Sometimes(0.5, iaa.Affine(
                translate_percent={"x": (-0.25, 0.25), "y": (-0.25, 0.25)},
                order=[0, 1],
                mode='constant',
                cval=(0, 255),
            )),
        ])

    @staticmethod
    def load_augmentation_aug_non_geometric():
        return iaa.Sequential([
            iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)),
            iaa.Sometimes(0.5, iaa.OneOf([
                iaa.GaussianBlur(sigma=(0.0, 3.0)),
                iaa.GaussianBlur(sigma=(0.0, 5.0))
            ])),
            iaa.Sometimes(0.5, iaa.MultiplyAndAddToBrightness(mul=(0.4, 1.7))),
            iaa.Sometimes(0.5, iaa.GammaContrast((0.4, 1.7))),
            iaa.Sometimes(0.5, iaa.Multiply((0.4, 1.7), per_channel=0.5)),
            iaa.Sometimes(0.5, iaa.MultiplyHue((0.4, 1.7))),
            iaa.Sometimes(0.5, iaa.MultiplyHueAndSaturation((0.4, 1.7), per_channel=True)),
            iaa.Sometimes(0.5, iaa.LinearContrast((0.4, 1.7), per_channel=0.5))
        ])
    
    def load_augmentation_aug_all(self):
        return iaa.OneOf([
            iaa.Sometimes(0.5, self.load_augmentation_aug_geometric()),
            iaa.Sometimes(0.5, self.load_augmentation_aug_non_geometric())
        ])
    
    def load_aug_by_name(self, aug_name="aug_all"):
        if not len(self.loaded_augmentation_name):
            self.loaded_augmentation_name = aug_name
            self.IMAGE_AUGMENTATION_SEQUENCE = self.augmentation_functions[aug_name]()
            
        return self.IMAGE_AUGMENTATION_SEQUENCE
