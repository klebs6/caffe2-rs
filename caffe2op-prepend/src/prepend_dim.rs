crate::ix!();

/**
  | Reshape the tensor by prepending a dimension
  | of fixed size and dividing the size of
  | the next dimension by that amount.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct PrependDimOp<Context> {
    storage:  OperatorStorage,
    context:  Context,
    dim_size: i64,
}

impl<Context> PrependDimOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            dim_size_(this->template GetSingleArgument<int64_t>("dim_size", 0)) 

        CAFFE_ENFORCE_GT(
            dim_size_, 0, "Argument dim_size must be greater than zero.");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
        auto* output = Output(0);

        CAFFE_ENFORCE(input.dim() > 0, "Input must be at least 1D.");
        CAFFE_ENFORCE(
            input.size(0) % dim_size_ == 0,
            "First dimension must be multiple of prepend_dim. Current first dimension: ",
            input.size(0));

        vector<int64_t> actual_new_shape(input.dim() + 1);
        actual_new_shape[0] = dim_size_;
        actual_new_shape[1] = input.size(0) / dim_size_;
        for (int i = 1; i < input.sizes().size(); ++i) {
          actual_new_shape[i + 1] = input.size(i);
        }
        output->Resize(actual_new_shape);

        if (output != &input) {
          // If we are not doing in-place computation, a copy is needed.
          context_.CopyItemsSameDevice(
              input.dtype(),
              input.numel(),
              input.raw_data(),
              output->raw_mutable_data(input.dtype()));
        }
        return true;
        */
    }
}
