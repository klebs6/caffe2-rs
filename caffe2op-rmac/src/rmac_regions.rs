crate::ix!();

/**
  | Computes a fixed-grid of RMAC region
  | coordinates at various levels as described
  | in https://arxiv.org/abs/1511.05879.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RMACRegionsOp<Context> {
    storage:  OperatorStorage,
    context:  Context,
    scales:   i32,
    overlap:  f32,
    num_rois: Tensor,
}

register_cpu_operator!{RMACRegions, RMACRegionsOp<CPUContext>}

num_inputs!{RMACRegions, 1}

num_outputs!{RMACRegions, 1}

inputs!{RMACRegions, 
    0 => ("X", "The input 4D tensor of shape NCHW.")
}

outputs!{RMACRegions, 
    0 => ("RMAC_REGIONS", "The output RMAC regions for all items in the batch. Tensor of shape (N x 5) following the ROIPoolOp format - each row is of the format (batch_index x1 y1 x2 y2) where x1, y1, x2, y2 are the region co-ordinates. Each region is repeated N times corresponding to each item in the batch.")
}

args!{RMACRegions, 
    0 => ("scales",  "Number of scales to sample regions at."),
    1 => ("overlap", "Overlap between consecutive regions.")
}

should_not_do_gradient!{RMACRegions}

impl<Context> RMACRegionsOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            scales_(this->template GetSingleArgument<int>("scales", 3)),
            overlap_(this->template GetSingleArgument<float>("overlap", 0.4f))
        */
    }
}
