def mask_tracking_lap(labels, min_tracklength):
    """
    LAP tracking of label images (3D-images (txy)).
    This script takes non-tracked label images (3D-images (txy)) and tracks them using lap track. The resulting
    label-image is saved as a tif file and tracking coordinates as csv.
    Args:
         labels: (ndarray) Label image series to be tracked
         min_tracklength: (int) Minimum track length to be considered for tracking
    Returns:
         new_labels: (ndarray) label image of tracked cells
         new_track_df: (pandas dataframe) tracking coordinates of tracked cells
    """
    import numpy as np
    import pandas as pd
    from laptrack import LapTrack
    from skimage.measure import regionprops_table

    regionprops = []
    for frame in range(labels.shape[0]):
        df = pd.DataFrame(regionprops_table(labels[frame, :, :], properties=["label", "centroid"]))
        df["frame"] = frame
        regionprops.append(df)
    regionprops_df = pd.concat(regionprops)
    # LapTrack cannot deal with missing/empty frames, so I include missing frame rows filled with nan here
    frames = pd.Series(range(labels.shape[0]), name='frame')
    regionprops_df = regionprops_df.merge(frames, how='right', on='frame')

    lt = LapTrack(gap_closing_max_frame_count=0)
    track_df, split_df, _ = lt.predict_dataframe(
        regionprops_df.copy(),
        coordinate_cols=["centroid-0", "centroid-1"],
        only_coordinate_cols=False,
    )

    # apply threshold on track length if needed
    if min_tracklength > 0:
        track_df = track_df.groupby('tree_id').filter(lambda x: len(x) >= min_tracklength).reset_index(drop=True)

    # create the new label image time series (tracked)
    new_labels = np.zeros_like(labels)
    for i, row in track_df.iterrows():
        frame = int(row["frame_y"])
        inds = labels[frame] == row["label"]
        new_labels[frame][inds] = int(row["track_id"]) + 1
    # new cleaned-up tracking dataframe that goes with the new label-image
    new_track_df = track_df[["frame_y", "centroid-0", "centroid-1", "track_id", "tree_id"]].copy()
    new_track_df = new_track_df.rename(
        columns={'frame_y': 'frame', 'centroid-0': 'centroid-y', 'centroid-1': 'centroid-x', 'tree_id': 'parental_id'})
    new_track_df['parental_id'] = new_track_df['parental_id'] + 1
    new_track_df['track_id'] = new_track_df['track_id'] + 1

    return new_labels, new_track_df
