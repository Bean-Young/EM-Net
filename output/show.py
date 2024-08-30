def from_nii_to_png(path,outputpath):
    import nibabel as nib
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    from skimage import filters
    from scipy.ndimage import binary_fill_holes

    image_data = np.load(path)
    image_data = image_data['array_data']
    image_data = image_data.astype(np.uint8)
    # import matplotlib.pyplot as plt
    # plt.imshow(image_data, cmap='gray')
    # plt.colorbar()  
    # plt.show()
    # image_data = np.squeeze(image_data, axis=(1, 3))
    # image_data = np.swapaxes(image_data, 1, 2)
    # image_data=image_data[:,:,1]

    #plt.imsave('1.jpg',image_data,cmap='gray')

    blurred = cv2.GaussianBlur(image_data, (5, 5), 0)
    thresh_val = filters.threshold_otsu(blurred)
    binary = blurred > thresh_val
    filled_image = binary_fill_holes(binary)
    final_image = (filled_image * 255).astype(np.uint8)
    kernel = np.ones((10,10), np.uint8)
    # closing = cv2.morphologyEx(final_image, cv2.MORPH_CLOSE, kernel)
    # opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    opening = cv2.morphologyEx(final_image, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    # closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
    output_path_smoothed = outputpath
    # closing = image_data
    #print(output_path_smoothed)
    plt.imsave(output_path_smoothed,closing,cmap='gray')