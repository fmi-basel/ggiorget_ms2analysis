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
    kernel = np.ones((int(radius/2), int(radius/2)), np.uint8)
    mask_blind = mask.copy()
    mask_blind = np.stack([erosion(mask_blind[i, ...], kernel) for i in range(mask_blind.shape[0])], axis=0)

    # blind mask at the given coordinates with the given radius
    for index, row in masked_coordinates.iterrows():
        t = round(row[0])
        y = round(row[1])
        x = round(row[2])
        mask_blind[t, y - radius:y + radius, x - radius:x + radius] = 0

    # pick random positions from the remaining coordinates
    random_pos_list = []
    for index, row in cell_ids.iterrows():
        y, x = np.where(mask_blind[row[0], :, :] == row[1])
        random_index = np.random.randint(len(x))
        random_pos_list.append([index, y[random_index], x[random_index]])
    random_pos = pd.DataFrame(random_pos_list, columns=['index','y', 'x']).set_index('index')
    random_pos = cell_ids.merge(random_pos, left_index=True, right_index=True)

    return random_pos
