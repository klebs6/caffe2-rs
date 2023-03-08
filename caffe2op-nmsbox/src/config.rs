crate::ix!();

register_cpu_operator!{
    BoxWithNMSLimit, 
    BoxWithNMSLimitOp<CPUContext>
}

num_inputs!{BoxWithNMSLimit, (2,3)}

num_outputs!{BoxWithNMSLimit, (3,6)}

inputs!{BoxWithNMSLimit, 
    0 => ("scores",                        "Scores, size (count, num_classes)"),
    1 => ("boxes",                         "Bounding box for each class, size (count, num_classes * 4). For rotated boxes, this would have an additional angle (in degrees) in the format [<optionaal_batch_id>, ctr_x, ctr_y, w, h, angle]. Size: (count, num_classes * 5)."),
    2 => ("batch_splits",                  "Tensor of shape (batch_size) with each element denoting the number of RoIs/boxes belonging to the corresponding image in batch. Sum should add up to total count of scores/boxes.")
}

outputs!{BoxWithNMSLimit, 
    0 => ("scores",                        "Filtered scores, size (n)"),
    1 => ("boxes",                         "Filtered boxes, size (n, 4). For rotated boxes, size (n, 5), format [ctr_x, ctr_y, w, h, angle]."),
    2 => ("classes",                       "Class id for each filtered score/box, size (n)"),
    3 => ("batch_splits",                  "Output batch splits for scores/boxes after applying NMS"),
    4 => ("keeps",                         "Optional filtered indices, size (n)"),
    5 => ("keeps_size",                    "Optional number of filtered indices per class, size (num_classes)")
}

args!{BoxWithNMSLimit, 
    0 => ("score_thresh",                  "(float) TEST.SCORE_THRESH"),
    1 => ("nms",                           "(float) TEST.NMS"),
    2 => ("detections_per_im",             "(int) TEST.DEECTIONS_PER_IM"),
    3 => ("soft_nms_enabled",              "(bool) TEST.SOFT_NMS.ENABLED"),
    4 => ("soft_nms_method",               "(string) TEST.SOFT_NMS.METHOD"),
    5 => ("soft_nms_sigma",                "(float) TEST.SOFT_NMS.SIGMA"),
    6 => ("soft_nms_min_score_thres",      "(float) Lower bound on updated scores to discard boxes"),
    7 => ("rotated",                       "bool (default false). If true, then boxes (rois and deltas) include angle info to handle rotation. The format will be [ctr_x, ctr_y, width, height, angle (in degrees)].")
}

should_not_do_gradient!{BoxWithNMSLimit}
