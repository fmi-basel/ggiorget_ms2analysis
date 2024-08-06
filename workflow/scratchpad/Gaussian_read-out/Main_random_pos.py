"""
After the initial tracking, additionally random position tracks are constructed, that cannot overlap with the 'real'
MS2 tracks.
====================
This script takes the 'real' MS2 tracks, corresponding labeled images, picks a number of random positions and constructs
traces based on those random positions.
The general workflow includes:
1. data loading
2. constructing random position tracks, including
    picking a random position, ignoring previous random positions and the real MS2 tracks with an additional radius
    around those positions
    based on this starting random position, construct track taking lateral cell movement and cell deformation into
    account
3. saving the random position tracks

author: Jana Tuennermann
"""

# Let's start with loading all the packages and modules you need for this script
import argparse
import os

import numpy as np
import pandas as pd
from skimage import io
from tqdm import tqdm

from gapclosing import gap_closing


def random_cell_coordinates(mask, cell_ids, masked_coordinates, radius=7):
    """
    Within a cell a random position per frame is picked.
    Given a mask and a corresponding list of cell ids, a random position per cell and time point is picked. Before
    picking a random position, from the mask, a roi is excluded around some given coordinates (normally corresponding
    to spot coordinates). The random position is picked from the remaining coordinates.

    Args:
         mask: (ndarray) label image from which to pick random positions.
         cell_ids (pd.dataframe) dataframe containing the cell ids and the corresponding frame numbers for which random
                                 positions should be picked
         masked_coordinates: (list) before picking random positions, certain coordinates should not be
                                    picked/are masked.
         radius: (int) size of the roi around the masked coordinates to be blinded for picking (square)

    Returns:
         random_pos: (pd.dataframe) dataframe containing cell ids, t, and randomly picked y, x coordinates
    """
    import numpy as np
    import pandas as pd
    from skimage.morphology import erosion

    # to not pick positions from the edges of the cell, the mask is eroded with the size of half the radius
    kernel = np.ones((int(radius / 2), int(radius / 2)), np.uint8)
    mask_blind = mask.copy()
    mask_blind = np.stack([erosion(mask_blind[i, ...], kernel) for i in range(mask_blind.shape[0])], axis=0)

    # blind mask at the given coordinates with the given radius
    for index, row in masked_coordinates.iterrows():
        y = round(row.iloc[1])
        x = round(row.iloc[2])
        mask_blind[:, y - radius:y + radius, x - radius:x + radius] = 0

    # pick random positions from the remaining coordinates
    random_pos_list = []
    for index, row in cell_ids.iterrows():
        y, x = np.where(mask_blind[row.iloc[0], :, :] == row.iloc[1])
        random_index = np.random.randint(len(x))
        random_pos_list.append([index, y[random_index], x[random_index]])
    random_pos = pd.DataFrame(random_pos_list, columns=['index', 'y', 'x']).set_index('index')
    random_pos = cell_ids.merge(random_pos, left_index=True, right_index=True)

    return random_pos


def main(tracks_path, segmentation_path, path_output, number_position=4):
    # Get the name for the movie (for naming convention later)
    images_filename = os.path.split(tracks_path)[1]

    # Check whether the output path for plots already exists, if not create it
    if not os.path.exists(path_output):
        # Create a new directory because it does not exist
        os.makedirs(path_output)

    # 1. ------- Data loading--------
    df_tracks = pd.read_csv(tracks_path)

    mask_labelimagename = images_filename.replace('_tracks.csv', '_label-image.tiff')
    mask_image = io.imread(os.path.join(segmentation_path, mask_labelimagename))
    mask_csvname = images_filename.replace('_tracks.csv', '_label-image_tracks.csv')
    mask_track = pd.read_csv(os.path.join(segmentation_path, mask_csvname))

    # 2. ------- Linking --------
    # generate a set of random positions, one per cell as a starting position. They are generated from the first frame a
    # cell is detected
    print('Constructing random pos tracks...')
    masked_pos = df_tracks.sort_values('frame').groupby('track_id').first().reset_index()

    staring_positions = []
    for value in range(number_position):
        random_pos = random_cell_coordinates(mask_image, masked_pos[['frame', 'track_id']], pd.concat(
            [df_tracks[['frame', 'track_id', 'y', 'x']], *staring_positions]))
        staring_positions.append(random_pos)

    # Actual construction of the track: based on these random start positions, the track is constructed by interpolating
    # 'the position in the next frame taking lateral cell movement and 'deformation' into account
    # Do it set by set
    track_ids = mask_track['track_id'].unique()
    df_random_tracks = []
    for set_index, set_positions in enumerate(staring_positions):
        print('Set:', set_index)
        # Here, semi-empty df with spot data and NaN for frames without spots, in here I write all the info I generate
        df_random_tracks_set = pd.merge(mask_track[['frame', 'track_id', 'parental_id']],
                                        set_positions[['frame', 'track_id', 'x', 'y']], how='left')

        # Gab filling, including lateral cell movement and cell deformation
        filled_gaps = {}
        for cell in tqdm(track_ids):
            # Do everything cell by cell, so get the cell label image, cell track, spot info etc
            df_cellspots = df_random_tracks_set[df_random_tracks_set['track_id'] == cell]
            cellmask_track = mask_track[mask_track['track_id'] == cell]
            cellmask_image = np.zeros(mask_image.shape)
            cellmask_image[:, :, :][mask_image == cell] = 1
            # actually fill the gabs
            df_cellspots = gap_closing(df_cellspots, 'x', 'y', cellmask_track, cellmask_image)
            # Store the result in the dictionary
            filled_gaps[cell] = df_cellspots

        # update the df with the filled gaps
        for cell, df_cellspots in filled_gaps.items():
            df_random_tracks_set.update(df_cellspots)
        df_random_tracks_set['set'] = set_index
        df_random_tracks.append(df_random_tracks_set)
    df_random_tracks = pd.concat(df_random_tracks)

    # 3. ------- Saving data --------
    df_random_tracks.to_csv(os.path.join(path_output, images_filename.replace('_tracks.csv', '_random-tracks.csv')),
                            index=False)
    print('Done :)')


if __name__ == "__main__":
    # Parse command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--tracks",
        type=str,
        default=None,
        required=True,
        help="Dataframe containing previous tracking results, requires absolute path",
    )
    parser.add_argument(
        "-is",
        "--input_segmentation",
        type=str,
        default=None,
        required=True,
        help="Directory in which segmentation images are saved, requires absolute path",
    )
    parser.add_argument(
        "-o",
        "--path_output",
        type=str,
        default=None,
        required=True,
        help="Path to output directory, requires absolute path",
    )
    parser.add_argument(
        "-np",
        "--number_positions",
        type=int,
        default=4,
        required=False,
        help="Number of random positions/tracks to be generated per cell. Default is 4",
    )
    args = parser.parse_args()

    main(tracks_path=args.tracks, segmentation_path=args.input_segmentation, path_output=args.path_output,
         number_position=args.number_positions)
