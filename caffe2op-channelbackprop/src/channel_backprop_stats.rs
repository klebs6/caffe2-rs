crate::ix!();

/**
  | Given an input tensor in NCHW format,
  | the gradient for the output of SpatialBN
  | and the per-channel mean and inverse
  | std var vectors for the input, computes
  | the per-channel bias and scale gradient
  | to be used during the backward pass for
  | subsequent spatial batch normalization
  | gradient calculation.
  | 
  | Typically, the results of this op are
  | subsequently reduced over multiple
  | devices to obtain statistics over a
  | larger batch size in cases where the
  | batch size for a single model copy is
  | too low to yield the full benefit of batch
  | normalization. The resulting bias
  | and scale can then be plugged back into
  | SpatialBNGradient to get results over
  | the larger batch size
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ChannelBackpropStatsOp<Context> {

    storage: OperatorStorage,
    context: Context,

    d_bias_scratch:  Tensor,
    d_scale_scratch: Tensor,
}

impl<Context> ChannelBackpropStatsOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
}

input_tags!{
    ChannelBackpropStatsOp {
        Input,
        SavedMean,
        SavedInvStddev,
        OutputGrad
    }
}

output_tags!{
    ChannelBackpropStatsOp {
        ScaleGrad,
        BiasGrad
    }
}

register_cpu_operator!{
    ChannelBackpropStats, 
    ChannelBackpropStatsOp<CPUContext>
}

num_inputs!{ChannelBackpropStats, 4}

num_outputs!{ChannelBackpropStats, 2}

inputs!{ChannelBackpropStats, 
    0 => ("X",            "The input 4-dimensional tensor of shape NCHW"),
    1 => ("mean",         "The mean saved from the forward pass as a 1-dimensional tensor of size C."),
    2 => ("inv_std",      "The saved inverse standard deviation as a 1-dimensional tensor of size C."),
    3 => ("output_grad",  "Gradient for the output layer of SpatialBN, here used as input because we are on the backward pass")
}

outputs!{ChannelBackpropStats, 
    0 => ("scale_grad",   "Gradient for the scale vector"),
    1 => ("bias_grad",    "Gradient for the bias vector")
}

should_not_do_gradient!{ChannelBackpropStats}
