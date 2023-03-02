crate::ix!();

pub struct BatchMomentsGradientOp<T, Context> {

    //USE_OPERATOR_CONTEXT_FUNCTIONS;

    storage: OperatorStorage,
    context: Context,

    order:   StorageOrder,
    phantom: PhantomData<T>,
}

num_inputs!{BatchMomentsGradient, 3}

num_outputs!{BatchMomentsGradient, 1}

register_cpu_operator!{BatchMomentsGradient, BatchMomentsGradientOp<f32, CPUContext>}

impl<T, Context> BatchMomentsGradientOp<T, Context> {
    
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
            const auto& dmu = Input(0);
        const auto& dvar = Input(1);
        const auto& X = Input(2);

        const int ndim = X.dim();
        const int N = X.dim32(0);
        const int C = order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1);
        const int HxW = X.numel() / (N * C);
        auto* dX = Output(0, X.sizes(), at::dtype<T>());
        const T* dmu_data = dmu.template data<T>();
        const T* dvar_data = dvar.template data<T>();
        const T* X_data = X.template data<T>();
        T* dX_data = dX->template mutable_data<T>();
        return order_ == StorageOrder::NCHW
            ? ComputeBatchMomentsGradientNCHW(
                  N, C, HxW, dmu_data, dvar_data, X_data, dX_data)
            : ComputeBatchMomentsGradientNHWC(
                  N, C, HxW, dmu_data, dvar_data, X_data, dX_data);
        */
    }
}

pub struct GetBatchMomentsGradient;

impl GetGradientDefs for GetBatchMomentsGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "BatchMomentsGradient",
            "",
            std::vector<std::string>{GO(0), GO(1), I(0)},
            std::vector<std::string>{GI(0)});
        */
    }
}

register_gradient!{BatchMoments, GetBatchMomentsGradient}
