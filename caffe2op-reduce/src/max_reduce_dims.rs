crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MaxReduceDimsOp<T,Context,const FIRSTDIMS: bool> {
    storage:         OperatorStorage,
    context:         Context,
    num_reduce_dims: i32,
    phantom:         PhantomData<T>,
}

impl<T,Context,const FIRSTDIMS: bool> MaxReduceDimsOp<T,Context,FIRSTDIMS> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            num_reduce_dims_( this->template GetSingleArgument<int32_t>("num_reduce_dim", 1))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);

        CAFFE_ENFORCE(
            num_reduce_dims_ >= 0 && num_reduce_dims_ <= X.dim(),
            "For N-dim input tensor, support num_reduce_dims in range [0, N].");

        const int rows = FIRSTDIMS ? X.size_to_dim(num_reduce_dims_)
                                   : X.size_to_dim(X.dim() - num_reduce_dims_);
        const int cols = FIRSTDIMS ? X.size_from_dim(num_reduce_dims_)
                                   : X.size_from_dim(X.dim() - num_reduce_dims_);

        vector<int64_t> output_shape;
        int start_index = FIRSTDIMS ? num_reduce_dims_ : 0;
        int end_index = FIRSTDIMS ? X.dim() : X.dim() - num_reduce_dims_;

        for (int i = start_index; i < end_index; ++i) {
          output_shape.push_back(X.sizes()[i]);
        }
        auto* Y = Output(0, output_shape, at::dtype<float>());
        float* out_data = Y->template mutable_data<float>();

        if (cols == 0 || rows == 0) {
          math::Set(Y->numel(), static_cast<float>(0), out_data, &context_);
          return true;
        }

        const int32_t* lengths_data = nullptr;
        if (InputSize() > 1) {
          const auto& lengths = Input(1);
          lengths_data = lengths.template data<int32_t>();
          CAFFE_ENFORCE(
              num_reduce_dims_ == 1,
              "Given lengths input, the number of reduce dimensions should be one.");
          const int batch_size = FIRSTDIMS ? cols : rows;
          CAFFE_ENFORCE(
              lengths.numel() == batch_size,
              "The size of lengths vector doesn't match the batch size.");
        }

        const float* data = X.template data<float>();
        Compute(rows, cols, data, lengths_data, out_data);
        return true;
        */
    }
}
