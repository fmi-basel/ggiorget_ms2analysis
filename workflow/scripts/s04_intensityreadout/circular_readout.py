def circular_readout(images, coordinates, diameter, size_ring, gapsize):
    """
    Given a movie and coordinates, this function reads out the intensity of the region of interest (ROI). The center ROI
    is circular (spot), surrounded by a ring-shaped second ROI (background), both can be seperated with a given
    gap-size.

    Args:
         images: (ndarray) movie from which the intensity is read out.
         coordinates (list) list with t,y,x coordinates of the center of the spot ROI
         diameter: (int) diameter of the spot ROI in pixel, should be an odd number
         size_ring: (int) size of the ring ROI in pixel
         gapsize: (int) separation of the two ROIs in pixel

    Returns:
         df_intensity: (pd.dataframe) dataframe containing t,y,x coordinates and intensity values (Mean, SD, Median,
         MAD, total integrated intensity)
    """

    import numpy as np
    import pandas as pd
    from scipy import stats

    # define masks of the spot and ring ROIs with given diameter, ring size and gapsize
    readout_mask_circle = np.lib.pad(create_circular_mask(hight=diameter, width=diameter),
                                     ((size_ring + gapsize, size_ring + gapsize),
                                      (size_ring + gapsize, size_ring + gapsize)), 'constant', constant_values='0')

    cutoff_mask = np.lib.pad(create_circular_mask(hight=diameter + gapsize * 2, width=diameter + gapsize * 2),
                             ((size_ring, size_ring), (size_ring, size_ring)), 'constant', constant_values='0')

    readout_mask_ring = create_circular_mask(hight=diameter + (size_ring + gapsize) * 2,
                                             width=diameter + (size_ring + gapsize) * 2) - cutoff_mask

    # pad the original movie with zeros to avoid index errors (too close to edge)
    pad_size = diameter + (size_ring + gapsize) * 2
    images_padded = np.lib.pad(images, ((0, 0), (pad_size, pad_size,), (pad_size, pad_size)), 'constant',
                               constant_values='0')

    # select ROI around the coordinates, mask accordingly and calculate the intensity
    df_intensity = []
    for row in np.arange(0, len(coordinates)):
        t = coordinates[row][0]
        y = coordinates[row][1]
        x = coordinates[row][2]

        roi = images_padded[int(t),
              round(y) + pad_size - int(diameter / 2 + size_ring + gapsize): round(y) + pad_size + int(
                  diameter / 2 + size_ring + gapsize) + 1,
              round(x) + pad_size - int(diameter / 2 + size_ring + gapsize): round(x) + pad_size + int(
                  diameter / 2 + size_ring + gapsize) + 1]

        mean_spot = np.mean(roi[np.nonzero(roi * readout_mask_circle)])
        sd_spot = np.std(roi[np.nonzero(roi * readout_mask_circle)])
        median_spot = np.median(roi[np.nonzero(roi * readout_mask_circle)])
        mad_spot = stats.median_abs_deviation(roi[np.nonzero(roi * readout_mask_circle)])
        integrated_spot = (roi * readout_mask_circle).sum()
        mean_background = np.mean(roi[np.nonzero(roi * readout_mask_ring)])
        sd_background = np.std(roi[np.nonzero(roi * readout_mask_ring)])
        median_background = np.median(roi[np.nonzero(roi * readout_mask_ring)])
        mad_background = stats.median_abs_deviation(roi[np.nonzero(roi * readout_mask_ring)])
        integrated_background = (roi * readout_mask_ring).sum()

        df_intensity.append(
            [t, y, x, integrated_spot, mean_spot, sd_spot, median_spot, mad_spot,
             integrated_background, mean_background, sd_background, median_background,
             mad_background])

    df_intensity = pd.DataFrame(df_intensity,
                                columns=['frame', 'y', 'x', 'integrated_spot', 'mean_spot',
                                         'sd_spot', 'median_spot', 'mad_spot', 'integrated_background',
                                         'mean_localbackground', 'sd_localbackground',
                                         'median_localbackground', 'mad_localbackground'])
    return df_intensity


def create_circular_mask(hight, width):
    """
    a circular or elliptical mask of a certain hight/witdh. The shape is centered
    Args:
         hight: (int) in pixel, needs to be uneven
         width: (int) in pixel, needs to be uneven

    Returns:
         binary mask: (array)
    """
    import numpy as np
    center = (int(width / 2), int(hight / 2))
    radius = min(center[0], center[1], width - center[0], hight - center[1])
    y, x = np.ogrid[:hight, :width]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask * 1
