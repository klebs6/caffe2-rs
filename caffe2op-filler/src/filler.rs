crate::ix!();

/**
  | FillerOp takes in either zero or one
  | input.
  | 
  | If the number of input is 1, the shape
  | will be identical to that of the input
  | at run time with optional additional
  | dimensions appended at the end as specified
  | by "extra_shape" argument. In that
  | case the "shape" parameter should not
  | be set.
  | 
  | If the number of inputs is 0, the full
  | shape must be provided via "shape" argument
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct FillerOp<Context> {
    storage:        OperatorStorage,
    context:        Context,
    shape:          Vec<i64>,
    extra_shape:    Vec<i64>,
    input_as_shape: bool,
}

impl<Context> FillerOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            shape_(this->template GetRepeatedArgument<int64_t>("shape")),
            extra_shape_(ToVectorint64_t(
                this->template GetRepeatedArgument<int>("extra_shape"))),
            input_as_shape_(
                this->template GetSingleArgument<bool>("input_as_shape", false)) 

        if (InputSize()) {
          if (shape_.size() != 0) {
            CAFFE_THROW(
                "Cannot set the shape argument and pass in an input at "
                "the same time");
          }
        } else {
          if (!extra_shape_.empty()) {
            CAFFE_THROW("Cannot set extra_shape when there is no input");
          }
          if (input_as_shape_) {
            CAFFE_THROW("An input must be given if input_as_shape is true");
          }
          if (shape_.size() == 0 &&
              this->template HasSingleArgumentOfType<int>("shape")) {
            CAFFE_THROW("Fill 'shape' argument was a scalar, list expected");
          }
        }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto* output = Operator<Context>::Output(0);
        if (InputSize()) {
          auto shape = vector<int64_t>{};
          if (input_as_shape_) {
            if (this->InputIsTensorType(0, CPU)) {
              // originally, shape input must be in CPU context
              auto& input = this->template Input<Tensor>(0, CPU);
              CAFFE_ENFORCE_EQ(
                  input.dim(),
                  1,
                  "When input_as_shape is true, the input must be a 1D tensor of "
                  "data type int64_t");
              CAFFE_ENFORCE(input.numel() > 0);
              auto* shape_data = input.template data<int64_t>();
              shape.insert(shape.end(), shape_data, shape_data + input.dim32(0));
            } else {
              // in ONNX case, we allow shape to be in CUDA context
              auto& input = Input(0);
              CAFFE_ENFORCE_EQ(
                  input.dim(),
                  1,
                  "When input_as_shape is true, the input must be a 1D tensor of "
                  "data type int64_t");
              CAFFE_ENFORCE(input.numel() > 0);
              auto* shape_data = input.template data<int64_t>();
              std::unique_ptr<int64_t[]> shape_data_copy =
                  std::make_unique<int64_t[]>(input.dim32(0));
              context_.template CopyToCPU<int64_t>(
                  input.dim32(0), shape_data, shape_data_copy.get());
              shape.insert(
                  shape.end(),
                  shape_data_copy.get(),
                  shape_data_copy.get() + input.dim32(0));
            }
          } else {
            auto& input = Input(0);
            shape.insert(shape.end(), input.sizes().begin(), input.sizes().end());
          }
          shape.insert(shape.end(), extra_shape_.begin(), extra_shape_.end());
          output->Resize(shape);
          shape_ = shape;
        } else {
          output->Resize(shape_);
        }
        return Fill(output);
        */
    }
}
