crate::ix!();

num_inputs!{PiecewiseLinearTransform, (1,4)}

num_outputs!{PiecewiseLinearTransform, 1}

inputs!{PiecewiseLinearTransform, 
    0 => ("predictions",           "2-D tensor (Tensor) of size (num_batches x num_classes) containing scores"),
    1 => ("bounds (optional)",     "See bounds in Arg. (bounds, slopes, intercepts) can be passed through either arg or input blobs."),
    2 => ("slopes (optional)",     "See slopes in Arg. (bounds, slopes, intercepts) can be passed through either arg or input blobs."),
    3 => ("intercepts (optional)", "See intercepts in Arg. (bounds, slopes, intercepts) can be passed through either arg or input blobs.")
}

outputs!{PiecewiseLinearTransform, 
    0 => ("transforms",            "2-D tensor (Tensor) of size (num_batches x num_classes) containing transformed predictions")
}

args!{PiecewiseLinearTransform, 
    0 => ("bounds",                "1-D vector of size (prediction_dimensions x (pieces+1)) contain the upper bounds of each piece of linear function. One special case is the first bound is the lower bound of whole piecewise function and we treat it the same as the left most functions. (bounds, slopes, intercepts) can be passed through either arg or input blobs."),
    1 => ("slopes",                "1-D vector of size (prediction_dimensions x pieces) containing the slopes of linear function"),
    2 => ("intercepts",            "1-D vector of size (prediction_dimensions x pieces) containing the intercepts of linear function"),
    3 => ("binary",                "If set true, we assume the input is a Nx1 or Nx2 tensor. If it is Nx1 tensor, it is positive predictions. If the input is Nx2 tensor, its first column is negative predictions and second column is positive and negative + positive = 1. We just need one group of piecewise linear functions for the positive predictions.")
}

input_tags!{
    PiecewiseLinearTransformOp {
        Predictions,
        Bounds,
        Slopes,
        Intercepts
    }
}
