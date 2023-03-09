crate::ix!();

/**
  | Given a partitioned tensor $T<N, D_1,
  | ..., D_n>$, where the partitions are
  | defined as ranges on its outer-most
  | (slowest varying) dimension $N$, return
  | a tensor $T<(N + 2 * padding\_width),
  | D_1, ...,
  | 
  | D_n>$ with paddings added to the start
  | and end of each range.
  | 
  | Optionally, different paddings can
  | be provided for beginning and end.
  | 
  | Paddings provided must be a tensor $T<D_1,
  | ...,
  | 
  | D_n>$. If no padding is provided, add
  | zero padding. If no lengths vector is
  | provided, add padding only once, at
  | the start and end of data.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sequence_ops.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct AddPaddingOp<Context> {

    storage:                    OperatorStorage,
    context:                    Context,

    start_padding_width:        i32,
    end_padding_width:          i32,

    /**
      | Scratch space required by the CUDA version
      | 
      | {Context::GetDeviceType()};
      |
      */
    lengths_prefix_sum_buffer:  Tensor,

    /// {Context::GetDeviceType()};
    lengths_prefix_sum:         Tensor,
}

num_inputs!{AddPadding, (1,4)}

num_outputs!{AddPadding, (1,2)}

inputs!{AddPadding, 
    0 => ("data_in",       "*(type: Tensor)* Input data ($T<N, D_1, ..., D_n>$)."),
    1 => ("lengths",       "*(type: Tensor`<int>`)* Number of elements in each range. sum(lengths) = N."),
    2 => ("start_padding", "*(type: Tensor`<int>`)* [OPTIONAL] Padding data for range start ($T<D_1, ..., D_n>$)."),
    3 => ("end_padding",   "*(type: Tensor`<int>`)* [OPTIONAL] Padding for range end. If not provided, `start_padding` is used ($T<D_1, ..., D_n>$).")
}

outputs!{AddPadding, 
    0 => ("data_out",    "*(type: Tensor)* Padded data tensor ($T<N + 2*padding_width, D_1, ..., D_n>$)."),
    1 => ("lengths_out", "*(type: Tensor`<int>`)* [OPTIONAL] Lengths for each padded range.")
}

args!{AddPadding, 
    0 => ("padding_width",     "*(type: int)* Number of copies of padding to add around each range."),
    1 => ("end_padding_width", "*(type: int)* [OPTIONAL] Specifies a different end-padding width. If this is not set, will use same as `padding_width`.")
}

impl<Context> AddPaddingOp<Context> {

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
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
    
        todo!();
        /*
            const auto& in = Input(0);
        CAFFE_ENFORCE_GE(in.dim(), 1);
        const int32_t outer_size = in.sizes()[0];
        const auto block_size = in.size_from_dim(1);

        // if no lengths is provided, assume it is a single full-span entry
        const int32_t* lengths_ptr = nullptr;
        int32_t lengths_size = 1;
        if (InputSize() > 1) {
          const auto& lengths = Input(1);
          lengths_ptr = lengths.template data<int32_t>();
          lengths_size = lengths.numel();
        }

        // fetch paddings
        // input_size == 2 : pad with zeros
        // input_size == 3 : start and end paddings are the same
        // input_size == 4 : different start and end paddings
        const T* padding_start_ptr = nullptr;
        const T* padding_end_ptr = nullptr;
        if (InputSize() >= 3) {
          auto& padding_start = Input(2);
          CAFFE_ENFORCE_EQ(block_size, padding_start.numel());
          padding_start_ptr = padding_start.template data<T>();
        }
        if (InputSize() == 4) {
          auto& padding_end = Input(3);
          CAFFE_ENFORCE_EQ(block_size, padding_end.numel());
          padding_end_ptr = padding_end.template data<T>();
        } else {
          padding_end_ptr = padding_start_ptr;
        }

        auto out_dims = in.sizes().vec();
        out_dims[0] += (startPaddingWidth_ + endPaddingWidth_) * lengths_size;
        auto* out = Output(0, std::move(out_dims), at::dtype<T>());

        const auto* in_ptr = in.template data<T>();
        auto* out_ptr = out->template mutable_data<T>();

        return MakePadding<T>(
            in_ptr,
            out_ptr,
            lengths_ptr,
            lengths_size,
            outer_size,
            padding_start_ptr,
            padding_end_ptr,
            block_size);
        */
    }
}
