def stardist_timeseries(imageseries, scale_factor=0.33, stardist_prob_thresh=0.48,
                        stardist_nms_thresh=0.3):
    """
    Label-images from timeseries using Stardist
    This script takes 2D-images (tyx) with a pseudo-nuclear staining and segments the nuclei using stardist and its
    standard '2D_versatile_fluo' model. It is possible to change the threshold parameters for segmentation or re-size
    the image before segmentation.

    Args:
         imageseries: (ndarray) Image series to perform segmentation for.
         scale_factor: (float) Resizes the image by this scaling factor.
         stardist_prob_thresh: (int) Probability threshold used by stardist. The default value for the
                                     '2D_versatile_fluo' model is stardist_prob_thresh = 0.48
         stardist_nms_thresh: (int) Overlap threshold used by stardist. The default value for the '2D_versatile_fluo'
                                    model is stardist_nms_thresh = 0.3

    Returns:
         labels: (ndarray) non-tracked label images of timeseries from segmentation
    """
    import numpy as np
    from csbdeep.utils import normalize
    from stardist.models import StarDist2D

    # creates a pretrained model for stardist, I use the '2D_versatile_fluo' model as default
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    # run stardist frame-by-frame:
    # Default threshold arguments for the '2D_versatile_fluo' model are prob_thresh=0.48 (probability threshold),
    # nms_thresh=0.3 (overlap threshold); This can be adjusted based on needs and should be adjusted for different
    # models.
    labels, _ = zip(*[
        model.predict_instances(normalize(imageseries[frame, ...]), prob_thresh=stardist_prob_thresh,
                                nms_thresh=stardist_nms_thresh, scale=scale_factor) for frame in
        range(imageseries.shape[0])])
    labels = np.asarray(labels)

    return labels
