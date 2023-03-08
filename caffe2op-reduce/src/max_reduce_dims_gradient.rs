crate::ix!();

///----------------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MaxReduceDimsGradientOp<T,Context,const FIRSTDIMS: bool> {
    storage:         OperatorStorage,
    context:         Context,
    num_reduce_dims: i32,
    phantom:         PhantomData<T>,
}

impl<T,Context,const FIRSTDIMS: bool> MaxReduceDimsGradientOp<T,Context,FIRSTDIMS> {

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
            auto& dY = Input(0);
        auto& X = Input(1);
        auto& Y = Input(2);

        auto* dX = Output(0, X.sizes(), at::dtype<float>());
        const int rows = FIRSTDIMS ? X.size_to_dim(num_reduce_dims_)
                                   : X.size_to_dim(X.dim() - num_reduce_dims_);
        const int cols = FIRSTDIMS ? X.size_from_dim(num_reduce_dims_)
                                   : X.size_from_dim(X.dim() - num_reduce_dims_);

        const float* dYdata = dY.template data<float>();
        const float* Xdata = X.template data<float>();
        const float* Ydata = Y.template data<float>();

        const int32_t* lengths_data = nullptr;
        if (InputSize() > 3) {
          const auto& lengths = Input(3);
          lengths_data = lengths.template data<int32_t>();
          CAFFE_ENFORCE(
              num_reduce_dims_ == 1,
              "Given lengths input, the number of reduce dimensions should be one.");
          const int batch_size = FIRSTDIMS ? cols : rows;
          CAFFE_ENFORCE(
              lengths.numel() == batch_size,
              "The size of lengths vector doesn't match the batch size.");
        }

        float* dXdata = dX->template mutable_data<float>();
        Compute(rows, cols, dYdata, Xdata, Ydata, lengths_data, dXdata);
        return true;
        */
    }
}
