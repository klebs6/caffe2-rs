crate::ix!();

/**
 | This op fills an output tensor with samples drawn
 | from a normal distribution specified by the mean
 | and standard deviation arguments. The output
 | tensor shape is specified by the *shape* argument.
 |
 | However, if *input_as_shape* is set to *true*,
 | then the *input* should be a 1D tensor containing
 | the desired output shape (the dimensions specified
 | in *extra_shape* will also be appended). In this
 | case, the *shape* argument should **not** be set.
 |
 | *Note: cannot set the shape argument and pass in
 | an input at the same time.*
 |
 | Github Links:
 | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h
 | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc
 */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct GaussianFillOp<T, Context> {
    base: FillerOp<Context>,
    mean: T,
    std:  T,
}

num_inputs!{GaussianFill, (0,1)}

num_outputs!{GaussianFill, 1}

inputs!{GaussianFill, 
    0 => ("input", "(Optional) 1D tensor specifying the shape of the output. Must be used with *input_as_shape=True*")
}

outputs!{GaussianFill, 
    0 => ("output", "Output tensor of random values drawn from a normal distribution. If the shape argument is set, this is the shape specified, and if the *input* exists and *input_as_shape=True*, it is the shape specified by the *input* tensor.")
}

args!{GaussianFill, 
    0 => ("mean",           "*(type: float; default: 0.)* Mean of the distribution to draw from."),
    1 => ("std",            "*(type: float; default: 1.)* Standard deviation of the distribution to draw from."),
    2 => ("shape",          "*(type: [int])* Desired shape of the *output* tensor."),
    3 => ("extra_shape",    "*(type: [int])* The additional dimensions appended at the end of the *shape* indicated by the input blob. Cannot set the *extra_shape* argument when there is no input blob."),
    4 => ("input_as_shape", "*(type: bool; default: False)* set to *True* to use the *input* as shape. First, input must be in CPU context.")
}

allow_inplace!{GaussianFill, vec![(0, 0)]}

tensor_inference_function!{GaussianFill, FillerTensorInference}

impl<T, Context> GaussianFillOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : FillerOp<Context>(std::forward<Args>(args)...),
            mean_(this->template GetSingleArgument<float>("mean", 0)),
            std_(this->template GetSingleArgument<float>("std", 1)) 

        DCHECK_GT(std_, 0) << "Standard deviation should be nonnegative.";
        */
    }
    
    #[inline] pub fn fill(&mut self, output: *mut Tensor) -> bool {
        
        todo!();
        /*
            math::RandGaussian<T, Context>(
            output->numel(),
            mean_,
            std_,
            output->template mutable_data<T>(),
            &context_);
        return true;
        */
    }
}
