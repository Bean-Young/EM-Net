def get_loss(tr_path,pr_path):
    import numpy as np
    import cv2
    import h5py
    import numpy as np
    from scipy.spatial.distance import cdist
    from scipy.ndimage import find_objects, label
    from skimage.measure import find_contours
    from numpy import percentile

    def compute_hd95(mask1, mask2):
        contours1 = find_contours(mask1, 0.5)
        contours2 = find_contours(mask2, 0.5)
        if contours1 and contours2:
            contour1 = contours1[0]
            contour2 = contours2[0]
            distances = cdist(contour1, contour2)
            hd95 = percentile(np.hstack((distances.min(axis=0), distances.min(axis=1))), 95)
            return hd95
        else:
            return None
    def dice_coefficient(image_true, image_pred):
        intersection = np.sum(image_true * image_pred)
        return (2. * intersection) / (np.sum(image_true) + np.sum(image_pred))

    def sensitivity(image_true, image_pred):
        TP = np.sum((image_true == 1) & (image_pred == 1))
        FN = np.sum((image_true == 1) & (image_pred == 0))
        return TP / float(TP + FN)


    def specificity(image_true, image_pred):
        TN = np.sum((image_true == 0) & (image_pred == 0))
        FP = np.sum((image_true == 0) & (image_pred == 1))
        return TN / float(TN + FP)


    def accuracy(image_true, image_pred):
        TP = np.sum((image_true == 1) & (image_pred == 1))
        TN = np.sum((image_true == 0) & (image_pred == 0))
        FP = np.sum((image_true == 0) & (image_pred == 1))
        FN = np.sum((image_true == 1) & (image_pred == 0))
        return (TP + TN) / float(TP + TN + FP + FN)


    def precision(image_true, image_pred):
        TP = np.sum((image_true == 1) & (image_pred == 1))
        FP = np.sum((image_true == 0) & (image_pred == 1))
        return TP / float(TP + FP)


    def f1_score(image_true, image_pred):
        p = precision(image_true, image_pred)
        r = sensitivity(image_true, image_pred)
        return 2 * (p * r) / (p + r)

    def mean_iou(image_true, image_pred):
        intersection = np.logical_and(image_true, image_pred)
        union = np.logical_or(image_true, image_pred)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    def resize_image(image, target_size=(224,224)):
        img = cv2.resize(image, dsize=target_size, interpolation=cv2.INTER_NEAREST)
        return img

    dataset_name = 'label'
    with h5py.File(tr_path, 'r') as file:
        gt = file[dataset_name][()]
        gt = gt[0, :, :]

    gt = (gt * 255).astype(np.uint8)

    img=cv2.imread(pr_path,cv2.IMREAD_GRAYSCALE)
    ret,img  = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    gt=gt.astype(bool)
    img=img.astype(bool)
    out={'dice_coefficient':dice_coefficient(gt,img),
         'sensitivity':sensitivity(gt,img),
         'specificity':specificity(gt,img),
         'accuracy':accuracy(gt,img),
         'precision':precision(gt,img),
         'f1_score':f1_score(gt,img),
         'mean_iou':mean_iou(gt,img)}
    return out