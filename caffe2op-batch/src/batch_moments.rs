crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct BatchMomentsOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    order:   StorageOrder,
    phantom: PhantomData<T>,
}

num_inputs!{BatchMoments, 1}

num_outputs!{BatchMoments, 2}

register_cpu_operator!{BatchMoments, BatchMomentsOp<f32, CPUContext>}

impl<T,Context> BatchMomentsOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
                 order_(StringToStorageOrder(
                         this->template GetSingleArgument<std::string>("order", "NCHW"))) 
                     CAFFE_ENFORCE_NE(order_, StorageOrder::UNKNOWN);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);

        const int ndim = X.dim();
        const int N = X.dim32(0);
        const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
        const int HxW = X.numel() / (N * C);
        auto* mu = Output(0, {C}, at::dtype<T>());
        auto* var = Output(1, {C}, at::dtype<T>());
        const T* X_data = X.template data<T>();
        T* mu_data = mu->template mutable_data<T>();
        T* var_data = var->template mutable_data<T>();
        return order_ == StorageOrder::NCHW
            ? ComputeBatchMomentsNCHW(N, C, HxW, X_data, mu_data, var_data)
            : ComputeBatchMomentsNHWC(N, C, HxW, X_data, mu_data, var_data);
        */
    }
}
