crate::ix!();

/**
  | Given DATA tensor with first dimension
  | N and SCALE vector of the same size N produces
  | an output tensor with same dimensions
  | as DATA.
  | 
  | Which consists of DATA slices.
  | 
  | i-th slice is divided by sqrt(SCALE[i])
  | elementwise. If SCALE[i] == 0 output
  | slice is identical to the input one (no
  | scaling)
  |
  */
#[USE_DISPATCH_HELPER]
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SquareRootDivideOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

register_cpu_operator!{SquareRootDivide, SquareRootDivideOp<CPUContext>}

num_inputs!{SquareRootDivide, 2}

num_outputs!{SquareRootDivide, 1}

allow_inplace!{SquareRootDivide, vec![(0, 0)]}

input_tags!{
    SquareRootDivideOp
    {
        Data,
        Scale
    }
}

register_gradient!{
    SquareRootDivide, 
    GetSquareRootDivideGradient
}

impl<Context> SquareRootDivideOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float>>::call(this, Input(DATA));
        */
    }
    
    #[inline] pub fn do_run_with_type<TData>(&mut self) -> bool {
    
        todo!();
        /*
            return DispatchHelper<TensorTypes2<float, int32_t, int64_t>, TData>::call(
            this, Input(SCALE));
        */
    }
    
    #[inline] pub fn do_run_with_type2<TData, TScale>(&mut self) -> bool {
    
        todo!();
        /*
            auto& data = Input(DATA);
        auto& scale = Input(SCALE);

        auto* Y = Output(0, data.sizes(), at::dtype<TData>());
        size_t batchSize = data.size(0);
        size_t exampleSize = data.size_from_dim(1);
        CAFFE_ENFORCE(batchSize == scale.size(0), batchSize, " != ", scale.size(0));
        auto* scalePtr = scale.template data<TScale>();
        auto* dataPtr = data.template data<TData>();
        auto* yPtr = Y->template mutable_data<TData>();
        for (auto i = 0U; i < batchSize; ++i) {
          auto scale = scalePtr[i];
          CAFFE_ENFORCE(scale >= 0, scale, " < 0");
          auto multiplier = scale == 0 ? 1.0 : 1 / std::sqrt(scale);
          math::Scale<float, TData, Context>(
              exampleSize,
              multiplier,
              dataPtr + i * exampleSize,
              yPtr + i * exampleSize,
              &context_);
        }
        return true;
        */
    }
}
