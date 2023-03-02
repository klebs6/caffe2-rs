crate::ix!();

///--------------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SquaredL2DistanceGradientOp<T, Context> {

    storage: OperatorStorage,
    context: Context,

    /**
      | Input: X, Y, dDistance;
      | 
      | Output: dX, dY
      |
      */
    phantom: PhantomData<T>,
}

register_cpu_operator!{
    SquaredL2DistanceGradient,
    SquaredL2DistanceGradientOp<f32, CPUContext>
}

num_inputs!{SquaredL2DistanceGradient, 3}

num_outputs!{SquaredL2DistanceGradient, 2}

pub struct GetSquaredL2DistanceGradient;

impl GetGradientDefs for GetSquaredL2DistanceGradient {
    
    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            return SingleGradientDef(
            "SquaredL2DistanceGradient", "",
            vector<string>{I(0), I(1), GO(0)},
            vector<string>{GI(0), GI(1)});
        */
    }
}

register_gradient!{
    SquaredL2Distance, 
    GetSquaredL2DistanceGradient
}

impl<T, Context> SquaredL2DistanceGradientOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
        auto& Y = Input(1);
        auto& dDistance = Input(2);

        int N = X.dim() > 0 ? X.dim32(0) : 1;
        int D = N > 0 ? X.numel() / N : 0;
        CAFFE_ENFORCE(X.dim() == Y.dim());
        for (int i = 0; i < X.dim(); ++i) {
          CAFFE_ENFORCE(X.dim32(i) == Y.dim32(i));
        }
        CAFFE_ENFORCE(dDistance.dim() == 1);
        CAFFE_ENFORCE(dDistance.dim32(0) == N);
        auto* dX = Output(0, X.sizes(), at::dtype<T>());
        auto* dY = Output(1, Y.sizes(), at::dtype<T>());
        math::Sub<T, Context>(
            X.numel(),
            X.template data<T>(),
            Y.template data<T>(),
            dX->template mutable_data<T>(),
            &context_);
        for (int i = 0; i < N; ++i) {
          math::Scale<T, T, Context>(
              D,
              dDistance.template data<T>() + i,
              dX->template data<T>() + i * D,
              dX->template mutable_data<T>() + i * D,
              &context_);
        }
        // The gradient of the other side is basically the negative.
        math::Scale<T, T, Context>(
            X.numel(),
            -1,
            dX->template data<T>(),
            dY->template mutable_data<T>(),
            &context_);
        return true;
        */
    }
}
