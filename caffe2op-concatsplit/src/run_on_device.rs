crate::ix!();

impl<Context> SplitOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            split_(this->template GetRepeatedArgument<int>("split")) 

        CAFFE_ENFORCE(
            !(OperatorStorage::HasArgument("axis") &&
              OperatorStorage::HasArgument("order")),
            "You shouldn't specify both the dim to split, and the order "
            "in the case of 4-D images.");
        if (OperatorStorage::HasArgument("axis")) {
          axis_ = this->template GetSingleArgument<int>("axis", -1);
          // only exists for computing the gradient of a Concat with 'add_axis'
          add_axis_ = this->template GetSingleArgument<int>("add_axis", 0);
        } else {
          axis_ = GetDimFromOrderString(
              this->template GetSingleArgument<string>("order", "NCHW"));
          add_axis_ = 0;
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
      int canonical_axis = input.canonical_axis_index(axis_);
      CAFFE_ENFORCE_LT(
          canonical_axis, input.dim(), "Axis not in input ndim range.");
      const int input_channels = input.dim32(canonical_axis);
      const int* axis_data;
      vector<int> equal_split;
      if (InputSize() == kSplitOpInputSize) {
        // We obtain split from the input tensor.
        CAFFE_ENFORCE_EQ(
            split_.size(),
            0,
            "If you set split with an input blob, do not pass in "
            "split in the argument.");
        auto& split_tensor = this->template Input<Tensor>(1, CPU);
        CAFFE_ENFORCE_EQ(split_tensor.numel(), OutputSize());
        axis_data = split_tensor.template data<int>();
      } else if (split_.size() == 0) {
        CAFFE_ENFORCE_EQ(
            input_channels % OutputSize(),
            0,
            "If you did not specify split explicitly, the number of "
            "input channels should be divisible by the output size.");
        equal_split.resize(OutputSize(), input_channels / OutputSize());
        axis_data = equal_split.data();
      } else {
        // We obtain split from the parameters.
        CAFFE_ENFORCE_EQ(
            split_.size(),
            OutputSize(),
            "The number of splits specified should be equal to the "
            "number of outputs.");
        axis_data = split_.data();
      }

      CAFFE_ENFORCE_EQ(
          add_axis_ ? OutputSize()
                    : std::accumulate(axis_data, axis_data + OutputSize(), 0),
          input_channels,
          "Sum of split dimensions do not match: should be ",
          input_channels);
      vector<int64_t> output_dims(input.sizes().vec());
      int before = 1, after = 1;
      for (int i = 0; i < canonical_axis; ++i) {
        before *= input.dim32(i);
      }
      for (int i = canonical_axis + 1; i < input.dim(); ++i) {
        after *= input.dim32(i);
      }
      if (add_axis_) {
        output_dims.erase(output_dims.begin() + canonical_axis);
      }
      size_t input_offset = 0;
      for (int i = 0; i < OutputSize(); ++i) {
        auto* output = Output(i);
        auto axis_dim = add_axis_ ? 1 : axis_data[i];
        if (!add_axis_) {
          output_dims[canonical_axis] = axis_data[i];
        }
        output->Resize(output_dims);
        math::CopyMatrix<Context>(
            input.itemsize(),
            before,
            axis_dim * after,
            static_cast<const char*>(input.raw_data()) + input_offset,
            input.dim32(canonical_axis) * after,
            output->raw_mutable_data(input.dtype()),
            axis_dim * after,
            &context_,
            input.dtype().copy());
        input_offset += axis_dim * after * input.itemsize();
      }
      return true;
        */
    }
}
