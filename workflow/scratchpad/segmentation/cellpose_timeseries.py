def cellpose_timeseries(imageseries, diameter=15, flow_threshold=-2, cellprob_threshold=0.8):
    """
    Label-images from timeseries using Cellpose
    This script takes 2D-image series (tyx) with a pseudo-nuclear staining and segments the nuclei using cellpose and
    its standard model 'nuclei'. It is possible to change the threshold parameters for segmentation.

    Args:
         imageseries: (ndarray) Image series to perform segmentation for
         diameter: (int) estimate diameter of cells/nuclei in pixel. The original model was trained on cells with a
                         diameter mean of 17 pixel. For automated estimation set diameter = None
         flow_threshold: (int) Flow threshold or model fit threshold. Cellpose uses this value for thresholding shape
                               recognition. It's default value is 0.4. Should be higher if not enough ROIs are
                               recognised and lower if too many are recognised.
         cellprob_threshold: (int) The default is cellprob_threshold=0.0. Decrease this threshold if cellpose is not
                                   returning as many ROIs as you’d expect. Similarly, increase this threshold if
                                   cellpose is returning too ROIs particularly from dim areas. (ranges from -6 to 6)

    Returns:
         mask: (ndarray) non-tracked label images of timeseries from segmentation
    """

    # Import packages
    import numpy as np
    from cellpose import models

    model = models.Cellpose(gpu=False, model_type='nuclei')

    segmented_series = []
    for frame in imageseries:
        # Segment the cells with Cellpose
        masks, flows, styles, diams = model.eval(frame, diameter=diameter, channels=[[0, 0]], do_3D=False,
                                                 flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold)

        # Append the segmentation mask to the list
        segmented_series.append(masks)

    # Convert the list of masks to a numpy array
    segmented_series = np.array(segmented_series)
    return segmented_series
