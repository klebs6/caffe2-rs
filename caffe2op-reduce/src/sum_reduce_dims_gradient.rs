crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SumReduceDimsGradientOp<Context,const FIRSTDIMS: bool,const NORMALIZE: bool> {

    storage: OperatorStorage,
    context: Context,

    /**
      | scratch space used for former version
      | of this reducer
      |
      */
    num_reduce_dims:  i32, //{Context::GetDeviceType()};

    shape:            Tensor,
}

impl<Context,const FIRSTDIMS: bool,const NORMALIZE: bool> 
SumReduceDimsGradientOp<Context, FIRSTDIMS, NORMALIZE> {
    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            num_reduce_dims_(
                this->template GetSingleArgument<int32_t>("num_reduce_dim", 1))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<int, long, float, double>>::call(
            this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
    
        todo!();
        /*
            auto& dY = Input(0);
        auto& input_1 = Input(1);

        vector<int64_t> dX_sizes;
        // In previous diff we changed the semantic: Input(1) was changed from
        // the shape of the input to the data tensor. This made the backward
        // computation incompatible with old models. To fix this, we check
        // the dimension and type of Input(1).
        if (input_1.dim() == 1 && input_1.template IsType<int64_t>()) {
          // Input(1) is the shape of the input
          shape_.CopyFrom(input_1);
          // Copy first dims
          dX_sizes = vector<int64_t>(
              shape_.template data<int64_t>(),
              shape_.template data<int64_t>() + shape_.numel());
        } else {
          // Input(1) is data tensor X
          dX_sizes = input_1.sizes().vec();
        }
        auto* dX = Output(0, dX_sizes, at::dtype<T>());

        const int rows = FIRSTDIMS ? dX->size_to_dim(num_reduce_dims_)
                                   : dX->size_to_dim(dX->dim() - num_reduce_dims_);
        const int cols = FIRSTDIMS
            ? dX->size_from_dim(num_reduce_dims_)
            : dX->size_from_dim(dX->dim() - num_reduce_dims_);

        const int32_t* lengths_data = nullptr;
        if (InputSize() > 2) {
          const auto& lengths = Input(2);
          lengths_data = lengths.template data<int32_t>();
          CAFFE_ENFORCE(
              num_reduce_dims_ == 1,
              "Given lengths input, the number of reduce dimensions should be one.");
          const int batch_size = FIRSTDIMS ? cols : rows;
          CAFFE_ENFORCE(
              lengths.numel() == batch_size,
              "The size of lengths vector doesn't match the batch size.");
        }

        const T* dYdata = dY.template data<T>();
        T* dXdata = dX->template mutable_data<T>();
        Compute<T>(rows, cols, dYdata, lengths_data, dXdata);
        return true;
        */
    }
}
