def get_output_image(data_path,mask_path,pre_path,output_path):
    from PIL import Image, ImageDraw
    import numpy as np
    from scipy.ndimage import binary_erosion
    import h5py

    def apply_continuous_edge_outline_to_images(image, mask, outline_color=(0, 255, 0, 255), thickness=4, final_size=(512, 512)):

        mask = mask.convert("L").resize(image.size)
        mask_array = np.array(mask) > 0


        structure = np.ones((thickness, thickness))
        eroded_mask_array = binary_erosion(mask_array, structure=structure)
        edge_mask_array = mask_array & (~eroded_mask_array)


        outline = Image.new("RGBA", image.size, (0, 0, 0, 0))
        outline_draw = ImageDraw.Draw(outline)

        for y, x in np.argwhere(edge_mask_array):
            outline_draw.ellipse((x - thickness // 2, y - thickness // 2, x + thickness // 2, y + thickness // 2), fill=outline_color)

        combined = Image.alpha_composite(image.convert("RGBA"), outline)

        combined_resized = combined.resize(final_size, Image.Resampling.LANCZOS)

        return combined_resized

    dataset_name = 'image'
    with h5py.File(data_path, 'r') as file:
        img_data = file[dataset_name][()]
        img_data = img_data.transpose((1, 2, 0))

    if img_data.dtype != np.uint8:
        img_data = (img_data * 255).astype(np.uint8)

    image = Image.fromarray(img_data)
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    dataset_name = "label"
    with h5py.File(mask_path, 'r') as file:
        img_data = file[dataset_name][()]
        img_data = img_data[0, :, :]

    img_data = (img_data * 255).astype(np.uint8)
    mask = Image.fromarray(img_data)
    mask = mask.convert("L")

    image=apply_continuous_edge_outline_to_images(image, mask,thickness=4)
    mask = Image.open(pre_path).convert("L")
    apply_continuous_edge_outline_to_images(image,mask,outline_color=(255,0,0,255),thickness=3).save(output_path,"PNG")
