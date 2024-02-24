def gap_closing(df_track, col_x, col_y, cellmask_track, cellmask_image):
    """
    Applies gap-closing between known positions of a one track (single cell) in 2D.
    For this, I take the first/last know position and interpolate the unknown position based on lateral cell movement
    and cell deformation. Cell rotation is not explicitly included, but is guessed by interpolating between two know
    positions if two or more are known. Otherwise, it is not considered at all.
    Args:
         df_track: (pd.dataframe) dataframe containing tracks with gabs to be interpolated
         col_x: (str) column names of x coordinates
         col_y: (str) column names of y coordinates
         cellmask_track: (pd.dataframe) dataframe containing tracks for cell movement
         cellmask_image: (ndarray) mask image/movie

    Returns:
         df_closedtrack: (pd.dataframe) dataframe containing gap closed tracks
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import distance_matrix
    from skimage import measure
    from scipy import stats

    # create lists with frames in which we detected spots or missing frames
    spot_frames = np.sort(list(df_track[df_track[col_y].notnull()]['frame'].astype(np.int64)))
    missing_frames = np.sort(list(df_track[df_track[col_y].isnull()]['frame'].astype(np.int64)))

    # get cell position in missing frames, save as list
    yx_cell_missing = cellmask_track[cellmask_track['frame'].isin(missing_frames)][
        ['centroid-y', 'centroid-x']].values.tolist()

    # angle between center of the cell and a detected spot. If no spot, the array is filled with NaN. Linearly
    # interpolate missing angles
    yx_spots_all = df_track[[col_y, col_x]].values.tolist()
    yx_cell_all = cellmask_track[['centroid-y', 'centroid-x']].values.tolist()
    position_angle = df_track[['frame']].copy()
    position_angle['angle'] = angle_between_two_points(yx_cell_all, yx_spots_all)
    position_angle = position_angle.interpolate(limit_direction='both')
    # save only missing angles into a list
    position_angle = position_angle[position_angle['frame'].isin(missing_frames)]
    position_angle = position_angle['angle'].values.tolist()

    # distance image for guessing cell deformation. Using thresholding, I can guess the relative distance from the
    # spot to the center of the cell.
    # distance image for whole movie
    mask_singlecell_dist = np.stack(
        [distance_transform(cellmask_image[i, ...], smoothing=True, sigma=5) for i in range(cellmask_image.shape[0])],
        axis=0)
    yx_spot_present = df_track[df_track[col_x].notnull()][[col_y, col_x]].values.tolist()
    # calculate the threshold percentile for frames where spot exist
    threshold_percentile = []
    for step, frame in enumerate(spot_frames):
        threshold_spot = mask_singlecell_dist[frame, round(yx_spot_present[step][0]), round(yx_spot_present[step][1])]
        threshold_percentile_spot = stats.percentileofscore(mask_singlecell_dist[frame, ...].flatten(),
                                                            threshold_spot)
        threshold_percentile.append([frame, threshold_percentile_spot])
    # Linearly interpolate missing threshold percentiles
    threshold_percentile = pd.DataFrame(threshold_percentile, columns=['frame', 'percentile']).merge(df_track['frame'],
                                                                                                     how='right')
    threshold_percentile = threshold_percentile.interpolate(limit_direction='both')
    # save only missing threshold percentiles in a list
    threshold_percentile = threshold_percentile[threshold_percentile['frame'].isin(missing_frames)]
    threshold_percentile = threshold_percentile['percentile'].values.tolist()

    # Here the actual gab filling: time point by time point calculate contour (distance to center), and angle vector to
    # estimate new position in the missing frames
    position_missing = []
    for step, timepoint in enumerate(missing_frames):
        try:
            threshold_intensity = np.percentile(mask_singlecell_dist[timepoint, ...].flatten(),
                                                threshold_percentile[step])
            contour = np.vstack(measure.find_contours(mask_singlecell_dist[timepoint, ...], threshold_intensity))
            position_vector = [newpoint_with_angle(position_angle[step],
                                                   yx_cell_missing[step][0],
                                                   yx_cell_missing[step][1], i) for i in range(1000)]
            distances = distance_matrix(position_vector, contour)
            min_index = np.where(distances == np.min(distances))[0][0]
            position = position_vector[min_index]
            position_missing.append([timepoint, *position])
        except ValueError:
            # use last know position if the extrapolation fails
            cell = df_track.iloc[0]['track_id']
            print(f'Gab closing exception occurred for cell {cell}, frame {timepoint}, linearly interpolated instead')
            position_missing.append([timepoint, np.nan, np.nan])

    position_missing = pd.DataFrame(position_missing, columns=['frame', col_y, col_x]).set_index('frame')

    # fill in the original df the NaNs with the calculated position, use frame as index to fill correctly.
    df_closedtrack = df_track.reset_index(level=0)
    df_closedtrack = df_closedtrack.set_index('frame').fillna(position_missing).reset_index().set_index('index')
    df_closedtrack = df_closedtrack.interpolate(limit_direction='both')

    return df_closedtrack


def distance_transform(mask, smoothing=True, sigma=25):
    """
    Apply euclidean distance transform to image
    Args:
         mask: (ndarray) binary mask image
         smoothing: (logical) apply gaussian after transformation
         sigma: (int) sigma of gaussian

    Returns:
         distance_img: (ndarray) distance transform
    """
    from skimage.filters import gaussian
    from scipy.ndimage import distance_transform_edt
    distance_img = distance_transform_edt(mask)
    if smoothing:
        distance_img = gaussian(distance_img, sigma=sigma)
    return distance_img


def angle_between_two_points(yx_center, yx_point):
    """
    calculate the angle between two points
    Args:
         yx_center: (list) every list element contains coordinates for y and x
         yx_point: (list) every list element contains coordinates for y and x

    Returns:
         angle: (list) angle between the two points in radians
    """

    import math
    spot_y = [y[0] for y in yx_point]
    spot_x = [x[1] for x in yx_point]
    center_y = [y[0] for y in yx_center]
    center_x = [x[1] for x in yx_center]
    angle = [math.atan2(spot_y[i] - center_y[i], spot_x[i] - center_x[i]) for i in range(len(spot_x))]
    return angle


def newpoint_with_angle(angle, point_y, point_x, radius):
    """
    Based on a given point and angle, calculates a new point at a given radius
    Args:
         angle: (float) angle in radians
         point_y: (float) in pixels
         point_x: (float) in pixels
         radius: (int)

    Returns:
         new point: (array) of new point position in y,x
    """
    import math
    new_point_y = point_y + radius * math.sin(angle)
    new_point_x = point_x + radius * math.cos(angle)
    return [new_point_y, new_point_x]
