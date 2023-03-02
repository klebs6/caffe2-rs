crate::ix!();

/**
  | This operator limits the given input
  | within an interval. The interval is
  | specified by the `min` and `max` arguments.
  | They default to numeric_limits::lowest()*
  | and numeric_limits::max()* respectively.
  | The clipping operation can be done in
  | an in-place fashion by using the same
  | output blob as the input blob.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/clip_op.cc
  |
  */
pub struct ClipOp<T, Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,
    min:     T,
    max:     T,
}

num_inputs!{Clip, 1}

num_outputs!{Clip, 1}

inputs!{Clip, 
    0 => ("X", "*(Tensor`<float>`)* Input tensor within range [*numeric_limits::lowest()*, *numeric_limits::max()*].")
}

outputs!{Clip, 
    0 => ("Y", "*(Tensor`<float>`)* Output tensor clipped within range [`min`, `max`].")
}

args!{Clip, 
    0 => ("min", "*(type: float)* Minimum value, under which element is replaced by min (default=*numeric_limits::lowest()*)."),
    1 => ("max", "*(type: float)* Maximum value, under which element is replaced by max (default=*numeric_limits::max()*).")
}

identical_type_and_shape!{Clip}

inherit_onnx_schema!{Clip}

allow_inplace!{Clip, vec![(0, 0)]}

impl<T,Context> ClipOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            min_(std::numeric_limits<T>::lowest()),
            max_(T::max) 

        if (HasArgument("min")) {
          min_ = static_cast<T>(this->template GetSingleArgument<float>("min", 0));
        }
        if (HasArgument("max")) {
          max_ = static_cast<T>(this->template GetSingleArgument<float>("max", 0));
        }
        */
    }
}
