crate::ix!();

/**
 | Fill the output tensor with float samples from
 | uniform distribution [`min`, `max`].
 |
 | - The range can be defined either by arguments or
 |   input blobs. `min` and `max` are inclusive.
 |
 |     - If the range is given by input blobs, you
 |       also need to give the shape as input.
 |
 |     - When the range is given as arguments, this
 |       operator enforces min <= max. When the range is
 |       given as inputs, the constraint is not enforced.
 |
 |     - When the range is given as inputs and max
 |       < min, the first dimension of the output is set to
 |       0. This behavior is allowed so that dynamically
 |       sampling indices into a dynamically sized tensor
 |       is possible.
 |
 | - The shape of the output can be given as argument
 |   or input.
 |
 | Github Links:
 | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.h
 | - https://github.com/caffe2/caffe2/blob/master/caffe2/operators/filler_op.cc
 |
 */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct UniformFillOp<T, Context> {
    base: FillerOp<Context>,
    min: T,
    max: T,
}

num_inputs!{UniformFill, vec![0, 1, 3]}

num_outputs!{UniformFill, 1}

inputs!{UniformFill, 
    0 => ("shape", "(*Tensor`<int>`*): 1-D tensor of the shape of the output, must be used with `input_as_shape` argument"),
    1 => ("min", "(*Tensor`<T>`*): scalar tensor containing minimum value, inclusive"),
    2 => ("max", "(*Tensor`<T>`*): scalar tensor containing maximum value, inclusive")
}

outputs!{UniformFill, 
    0 => ("output", "(*Tensor`<T>`*): filled output tensor")
}

args!{UniformFill, 
    0 => ("min", "(*T*): minimum value, inclusive"),
    1 => ("max", "(*T*): maximum value, inclusive"),
    2 => ("shape", "(*Tuple(int)*): shape of the output, do not set when `input_as_shape`=1"),
    3 => ("input_as_shape", "(*int*): set to 1 to use the first input as shape; 
        `shape` input must be in CPU context")
}

allow_inplace!{UniformFill, vec![(0, 0)]}

tensor_inference_function!{UniformFill, FillerTensorInference }

impl<T, Context> UniformFillOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : FillerOp<Context>(std::forward<Args>(args)...),
            min_(this->template GetSingleArgument<T>("min", 0)),
            max_(this->template GetSingleArgument<T>("max", 1)) 

        if (InputSize() == 3) {
          CAFFE_ENFORCE(
              !this->template HasSingleArgumentOfType<T>("min"),
              "Cannot set both min arg and min input blob");
          CAFFE_ENFORCE(
              !this->template HasSingleArgumentOfType<T>("max"),
              "Cannot set both max arg and max input blob");
        } else {
          CAFFE_ENFORCE_LT(
              min_, max_, "Max value should be bigger than min value.");
        }
        */
    }
    
    #[inline] pub fn fill(&mut self, output: *mut Tensor) -> bool {
        
        todo!();
        /*
            T min = min_;
        T max = max_;
        if (InputSize() == 3) {
          CAFFE_ENFORCE_EQ(1, Input(1).numel(), "min blob must be scalar");
          CAFFE_ENFORCE_EQ(1, Input(2).numel(), "max blob must be scalar");
          min = *Input(1).template data<T>();
          max = *Input(2).template data<T>();
          if (min > max) {
            auto shape = output->sizes().vec();
            shape[0] = 0;
            output->Resize(shape);
            output->template mutable_data<T>();
            return true;
          }
        }
        math::RandUniform<T, Context>(
            output->numel(),
            min,
            max,
            output->template mutable_data<T>(),
            &context_);
        return true;
        */
    }
}
