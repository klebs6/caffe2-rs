crate::ix!();

/**
  | Merge first two dimensions in a single
  | dimension with size dim(0) * dim(1).
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MergeDimOp<Context> {
    storage:  OperatorStorage,
    context:  Context,
    dim_size: i64,
}

impl<Context> MergeDimOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& input = Input(0);
        auto* output = Output(0);

        CAFFE_ENFORCE(input.dim() > 1, "Input must be at least 2D.");

        vector<int64_t> actual_new_shape(input.dim() - 1);
        actual_new_shape[0] = input.size(0) * input.size(1);
        for (int i = 1; i < input.sizes().size() - 1; ++i) {
          actual_new_shape[i] = input.size(i + 1);
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
