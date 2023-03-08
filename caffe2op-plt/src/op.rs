crate::ix!();

/**
  | PiecewiseLinearTransform takes inputs
  | -- predictions, a 2-D or 1-D tensor (Tensor)
  | of size (batch_size x prediction_dimensions).
  | 
  | The piecewise linear functions are
  | stored in bounds, slopes and intercepts.
  | The output tensor has the same shape
  | of input `predictions` and contains
  | the predictions transformed by the
  | piecewise linear functions.
  | 
  | Each column of predictions has its own
  | piecewise linear transformation functions.
  | 
  | Therefore the size of piecewise function
  | parameters are pieces x prediction_dimensions,
  | except for binary predictions where
  | only the positive prediction needs
  | them.
  | 
  | -----------
  | @note
  | 
  | in each piece, low bound is excluded
  | while high bound is included. Also the
  | piecewise linear function must be continuous.
  | 
  | Notes
  | 
  | - If the input is binary predictions
  | (Nx2 or Nx1 tensor), set the binary arg
  | to true so that one group of piecewise
  | linear functions is needed (see details
  | below).
  | 
  | - The transform parameters (bounds,
  | slopes, intercepts) can be passed either
  | through args or through input blobs.
  | 
  | - If we have multiple groups of piecewise
  | linear functions, each group has the
  | same number of pieces.
  | 
  | - If a prediction is out of the bounds,
  | it is capped to the smallest or largest
  | bound.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct PiecewiseLinearTransformOp<T, Context> {
    storage:             OperatorStorage,
    context:             Context,
    binary:              bool,
    bounds_from_arg:     Vec<T>,
    slopes_from_arg:     Vec<T>,
    intercepts_from_arg: Vec<T>,
    bounds_device:       Tensor, //{Context::GetDeviceType()};
    intercepts_device:   Tensor, //{Context::GetDeviceType()};
    slopes_device:       Tensor, //{Context::GetDeviceType()};
    gpu_copied:          bool,   // = false;

    /**
      | If true, the piecewise linear functions
      | are passed through args, otherwise,
      | they are passed through Input blobs.
      |
      */
    transform_param_from_arg: bool,
}
