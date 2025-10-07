import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float
from skimage.filters import gaussian
from skimage.morphology import disk, closing, remove_small_objects
from skimage.measure import label, regionprops_table
from skimage.segmentation import clear_border

def detect_dark_on_flatfield(img_path='image_2_prakriti_cropped.png', pixel_size_um=1):
    # ---- Tunables ----
    bg_sigma     = 40      # Gaussian sigma for background (vignette)
    dark_percent = 12      # keep darkest X% pixels
    min_area_px  = 120     # remove blobs smaller than this
    close_r      = 2       # morphological closing radius
    rim_frac     = 0.92    # ignore vignetted rim
    # ------------------

    # Read & grayscale
    I0 = io.imread(img_path)
    if I0.ndim == 3:
        if I0.shape[2] == 4:  # RGBA image
            I0 = I0[..., :3]  # Drop alpha channel
        I0 = color.rgb2gray(I0)
    I0 = img_as_float(I0)

    # --- Flat-field only ---
    bg = gaussian(I0, sigma=bg_sigma)
    F  = I0 - bg
    F  = F - np.min(F)
    F  = F / (np.max(F) + np.finfo(float).eps)

    # --- Threshold darkest part of FLAT-FIELD only ---
    th = np.percentile(F, dark_percent)
    BW = F <= th

    # Optional: ignore vignette rim
    h, w = F.shape
    xg, yg = np.meshgrid(np.arange(w) - w/2, np.arange(h) - h/2)
    roi = (np.sqrt(xg**2 + yg**2) <= min(h, w) * 0.5 * rim_frac)
    BW[~roi] = 0

    # Clean mask
    BW = closing(BW, disk(close_r))
    BW = np.array(BW, dtype=bool)
    BW = clear_border(BW)
    BW = remove_small_objects(BW, min_size=min_area_px)

    # --- Label & measure properties ---
    label_img = label(BW)
    props = regionprops_table(
        label_img, 
        properties=['area', 'centroid', 'bbox', 'equivalent_diameter', 'major_axis_length', 'minor_axis_length']
    )
    T = pd.DataFrame(props)

    if not T.empty:
        T['CentroidX_px'] = T['centroid-1']
        T['CentroidY_px'] = T['centroid-0']
        T['CentroidX_um'] = T['CentroidX_px'] * pixel_size_um
        T['CentroidY_um'] = T['CentroidY_px'] * pixel_size_um
        T['EquivDiameter_um']   = T['equivalent_diameter'] * pixel_size_um
        T['MajorAxisLength_um'] = T['major_axis_length'] * pixel_size_um
        T['MinorAxisLength_um'] = T['minor_axis_length'] * pixel_size_um
        T['Area_um2']           = T['area'] * (pixel_size_um**2)

    # ---- Plots ----
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    axs = axs.ravel()
    axs[0].imshow(I0, cmap='gray')
    axs[0].set_title('1) Original')
    axs[1].imshow(bg, cmap='gray')
    axs[1].set_title('2) Estimated background')
    axs[2].imshow(F, cmap='gray')
    axs[2].set_title('3) Flat-fielded')
    axs[3].imshow(BW, cmap='gray')
    axs[3].set_title(f'4) Darkest {dark_percent}% + cleanup')

    # Overlay detections
    axs[4].imshow(I0, cmap='gray')
    if not T.empty:
        for _, row in T.iterrows():
            minr, minc, maxr, maxc = int(row['bbox-0']), int(row['bbox-1']), int(row['bbox-2']), int(row['bbox-3'])
            rect = plt.Rectangle((minc, minr), maxc-minc, maxr-minr, edgecolor='r', facecolor='none', linewidth=1.5)
            axs[4].add_patch(rect)
            axs[4].plot(row['CentroidX_px'], row['CentroidY_px'], 'g+', markersize=7, linewidth=1.2)
    axs[4].set_title(f'5) Detections: {len(T)}')

    # Darkness score for intuition
    score = (th - F) * BW
    score[score < 0] = 0
    axs[5].imshow(score, cmap='gray')
    axs[5].set_title('6) Darkness score (masked)')
    plt.tight_layout()
    plt.show()

    # ---- Stats in console ----
    print(f'Dark spots detected: {len(T)}')
    if not T.empty:
        print(f"Median area = {T['area'].median():.1f} px² ({T['Area_um2'].median():.2f} µm²)")
        print(f"Median diameter = {T['equivalent_diameter'].median():.2f} px ({T['EquivDiameter_um'].median():.2f} µm)")

    # ---- Export centroids + dimensions ----
    if not T.empty:
        print('Crystal positions and dimensions:')
        print(T[['CentroidX_um','CentroidY_um','EquivDiameter_um','MajorAxisLength_um','MinorAxisLength_um','Area_um2']])
        T.to_csv('crystal_measurements.csv', index=False)
        print('Results exported to crystal_measurements.csv')

if __name__ == '__main__':
    detect_dark_on_flatfield()