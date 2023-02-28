crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/Reduction.h]

/**
  | NB: Keep this in sync with Reduction class in
  | torch/nn/_reduction.py
  |
  | These constants control the reduction behavior
  | of loss functions.
  |
  | Ideally, this would be a scoped enum, but jit
  | doesn't support that
  */
pub enum Reduction {

    /// Do not reduce
    None,             

    /// (Possibly weighted) mean of losses
    Mean,             

    /// Sum losses
    Sum,              
    END
}
