crate::ix!();

/**
  | Remove padding around the edges of each
  | segment of the input data. This is the
  | reverse operation of **AddPadding**,
  | and uses the same arguments and conventions
  | for input and output data format.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sequence_ops.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RemovePaddingOp<Context> {

    storage:                    OperatorStorage,
    context:                    Context,

    start_padding_width:        i32,
    end_padding_width:          i32,

    /// Scratch space required by the CUDA version
    lengths_prefix_sum_buffer:  Tensor, // {Context::GetDeviceType()};
    lengths_prefix_sum:         Tensor, // {Context::GetDeviceType()};
}

num_inputs!{RemovePadding, (1,2)}

num_outputs!{RemovePadding, (1,2)}

inputs!{RemovePadding, 
    0 => ("data_in", "Input tensor ($T<N, D_1, ..., D_n>$)."),
    1 => ("lengths", "*(type: Tensor`<int>`)* Number of elements in each range. sum(lengths) = N. If not provided, considers all data as a single segment.")
}

outputs!{RemovePadding, 
    0 => ("data_out", "*(type: Tensor)* Padded data tensor ($T<N + 2*padding_width, D_1, ..., D_n>$)."),
    1 => ("lengths_out", "*(type: Tensor`<int>`)* [OPTIONAL] Lengths for each padded range.")
}

args!{RemovePadding, 
    0 => ("padding_width", "*(type: int)* Outer-size of padding to remove around each range."),
    1 => ("end_padding_width", "*(type: int)* [OPTIONAL] Specifies a different end-padding width. If this is not set, will use same as `padding_width`.")
}

impl<Context> RemovePaddingOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            startPaddingWidth_( this->template GetSingleArgument<int>("padding_width", 1)),
            endPaddingWidth_( this->template GetSingleArgument<int>("end_padding_width", -1)) 

        CAFFE_ENFORCE_GE(startPaddingWidth_, 0);
        if (endPaddingWidth_ < 0) {
          endPaddingWidth_ = startPaddingWidth_;
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (startPaddingWidth_ == 0 && endPaddingWidth_ == 0) {
          Output(0)->CopyFrom(Input(0), true /*async*/);
          if (OutputSize() == 2) {
            Output(1)->CopyFrom(Input(1), true /*async*/);
          }
          return true;
        }
        return DispatchHelper<TensorTypes<float, double, int, int64_t, bool>>::call(
            this, Input(0));
        */
    }
}
