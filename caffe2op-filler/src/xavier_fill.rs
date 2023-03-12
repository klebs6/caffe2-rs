crate::ix!();

/**
  | This op fills an output tensor with values
  | sampled from a uniform distribution
  | with the range determined by the desired
  | shape of the output.
  | 
  | Rather, than specifying the range of
  | values manually, the novelty of Xavier
  | Fill is that it automatically scales
  | the range of the distribution it draws
  | from based on the size of the desired
  | output tensor.
  | 
  | For more information check out the paper
  | [Understanding the difficulty of training
  | deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).
  | The output tensor shape is specified
  | by the *shape* argument.
  | 
  | However, if *input_as_shape* is set
  | to *true*, then the *input* should be
  | a 1D tensor containing the desired output
  | shape (the dimensions specified in
  | *extra_shape* will also be appended).
  | In this case, the *shape* argument should
  | **not** be set.
  | 
  | -----------
  | @note
  | 
  | Do not set the shape argument and pass
  | in an input at the same time.*
  | 
  | Github Links:
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h
  | 
  | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct XavierFillOp<T, Context> {
    base:    FillerOp<Context>,
    phantom: PhantomData<T>,
}

num_inputs!{XavierFill, (0,1)}

num_outputs!{XavierFill, 1}

inputs!{XavierFill, 
    0 => ("input", "(Optional) 1D tensor specifying the shape of the output. Must be used with *input_as_shape=True*")
}

outputs!{XavierFill, 
    0 => ("output", "Output tensor of random values drawn from an automatically scaled uniform distribution, based on the size of the output tensor. If the shape argument is set, this is the shape specified by the shape argument, and if the *input* exists and *input_as_shape=True*, it is the shape specified by the *input* tensor.")
}

args!{XavierFill, 
    0 => ("shape", "*(type: [int])* Desired shape of the *output* tensor."),
    1 => ("extra_shape", "*(type: [int])* The additional dimensions appended at the end of the *shape* indicated by the input blob. Cannot set the *extra_shape* argument when there is no input blob."),
    2 => ("input_as_shape", "*(type: bool; default: False)* set to *True* to use the *input* as shape. First, input must be in CPU context.")
}

allow_inplace!{XavierFill, vec![(0, 0)]}

tensor_inference_function!{XavierFill, FillerTensorInference}

impl<T,Context> XavierFillOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : FillerOp<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn fill(&mut self, output: *mut Tensor) -> bool {
        
        todo!();
        /*
            const int fan_in = output->numel() / output->dim32(0);
        T scale = std::sqrt(T(3) / fan_in);
        math::RandUniform<T, Context>(
            output->numel(),
            -scale,
            scale,
            output->template mutable_data<T>(),
            &context_);
        return true;
        */
    }
}
