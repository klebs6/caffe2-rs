crate::ix!();

impl<InputTypes,Context,Functor,TypeMap> PowOp<InputTypes,Context,Functor,TypeMap> {
    
    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(bool, "broadcast", enable_broadcast_, 0),
            OP_SINGLE_ARG(int, "axis", axis_, -1),
            OP_SINGLE_ARG(string, "axis_str", axis_str_, ""),
            OP_SINGLE_ARG(string, "order", order_, "NCHW"),
            functor_() 

        if ((InputSize() == 1) && HasArgument("exponent")) { // UnaryElementwiseOp
          exponent_ = this->template GetSingleArgument<float>(
              "exponent", 0); // based on pow_ops.h
        } else if (InputSize() == 2) { // BinaryElementwiseOp
          // Figure out the correct axis to use.
          if (enable_broadcast_) {
            if (axis_ != -1) {
              // Get axis from an explicit axis argument.
              CAFFE_ENFORCE_EQ(
                  axis_str_.size(),
                  0U,
                  "Args axis and axis_str cannot be used simultaneously.");
            } else if (axis_str_.size()) {
              // Get the axis index semantically.
              CAFFE_ENFORCE_EQ(
                  axis_str_.size(), 1U, "Unsupported axis string", axis_str_);
              size_t semantic_axis_ = order_.find(axis_str_);
              CAFFE_ENFORCE_NE(
                  semantic_axis_,
                  string::npos,
                  "Unrecognizable axis string ",
                  axis_str_,
                  " from order string ",
                  order_);
              axis_ = semantic_axis_;
            }
          } else {
            CAFFE_ENFORCE(
                axis_ == -1 && axis_str_.empty(),
                "Do not specify axis or axis_str if broadcast is not enabled.");
          }
        } else {
          CAFFE_THROW(
              "Only a tensor with an argument or two input tensors are supported as input to pow operator.");
        }
        */
    }
}
