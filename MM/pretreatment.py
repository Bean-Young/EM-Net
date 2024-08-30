def pretreatment_for_fields(path):
    import cv2
    import numpy as np

    def DHR(src):
        grayScale = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
        ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        dst = cv2.inpaint(src, thresh2, 1, cv2.INPAINT_TELEA)
        return dst


    def increase_brightness(image, value=30):
        if value < -255:
            value = -255
        elif value > 255:
            value = 255

        adjusted_image = image.astype(np.float32)
        adjusted_image += value
        adjusted_image = np.clip(adjusted_image, 0, 255)
        brightened_image = adjusted_image.astype(np.uint8)
        return brightened_image

    def enhance_contrast(image, clip_limit=4.0):
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab_image)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        enhanced_image = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        return enhanced_image


    def color_normalization(image):
        channels = cv2.split(image)
        normalized_channels = [cv2.normalize(c, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) for c in channels]
        normalized_image = cv2.merge(normalized_channels)
        return normalized_image

    image=cv2.imread(path)
    image=cv2.resize(image, (512, 512),interpolation = cv2.INTER_LINEAR)
    image=DHR(image)
    image=enhance_contrast(image)
    image=increase_brightness(image)
    image=color_normalization(image)
    return image
