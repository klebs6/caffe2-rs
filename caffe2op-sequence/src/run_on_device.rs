crate::ix!();

impl<Context> GatherPaddingOp<Context> {

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
          Output(0)->Resize(std::vector<int64_t>(0));
          auto output_0_data = Output(0)->template mutable_data<int64_t>();
          // TODO(zhengxq): as suggested by salex@, change this to a loop.
          math::Set<int64_t, Context>(
              Output(0)->numel(), 0, output_0_data, &context_);
          if (OutputSize() == 2) {
            Output(1)->Resize(std::vector<int64_t>(0));
            auto output_1_data = Output(1)->template mutable_data<int64_t>();
            math::Set<int64_t, Context>(
                Output(1)->numel(), 0, output_1_data, &context_);
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
        const auto pad_width = startPaddingWidth_ + endPaddingWidth_;

        // if no lengths is provided, assume it is a single full-span entry
        const int32_t* lengths_ptr = &outer_size;
        int64_t lengths_size = 1;
        if (InputSize() > 1) {
          const auto& lengths = Input(1);
          lengths_ptr = lengths.template data<int32_t>();
          lengths_size = lengths.numel();
        }
        std::vector<int64_t> padShape(in.sizes().begin() + 1, in.sizes().end());
        // output will contain accumulator over paddings
        Output(0)->Resize(padShape);
        T* padding_start_ptr = Output(0)->template mutable_data<T>();
        math::Set<T, Context>(block_size, 0.0, padding_start_ptr, &context_);

        // if no end_padding is provided, assume it's the same as start_padding
        T* padding_end_ptr = padding_start_ptr;
        if (OutputSize() == 2) {
          Output(1)->Resize(padShape);
          padding_end_ptr = Output(1)->template mutable_data<T>();
          math::Set<T, Context>(block_size, 0.0, padding_end_ptr, &context_);
        }
        GatherPadding<T>(
            outer_size,
            lengths_size,
            block_size,
            pad_width,
            in.template data<T>(),
            lengths_ptr,
            padding_start_ptr,
            padding_end_ptr);
        return true;
        */
    }
}
