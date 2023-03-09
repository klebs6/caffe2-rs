crate::ix!();

impl<Context> ShapeOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            axes_(OperatorStorage ::GetRepeatedArgument<int>("axes"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& data = Input(DATA);

        int numDims = data.dim();
        int numAxes = axes_.size();
        if (numAxes == 0) {
          auto* output = Output(0, {numDims}, at::dtype<int64_t>());
          int64_t* output_data = output->template mutable_data<int64_t>();
          context_.CopyBytesSameDevice(
              numDims * sizeof(int64_t), data.sizes().data(), output_data);
          return true;
        }

        auto* output = Output(0, {numAxes}, at::dtype<int64_t>());
        auto src = reinterpret_cast<const char*>(data.sizes().data());
        auto out = reinterpret_cast<char*>(output->template mutable_data<int64_t>());
        for (int i = 0; i < numAxes; i++) {
          auto axis = axes_[i];
          CAFFE_ENFORCE_LT(axis, numDims, "Axis out of range");
          CAFFE_ENFORCE_GE(axis, 0, "Each axis should be non-negative");
          context_.CopyBytesSameDevice(
              sizeof(int64_t), src + axis * sizeof(int64_t), out);
          out += sizeof(int64_t);
        }
        return true;
        */
    }
}
