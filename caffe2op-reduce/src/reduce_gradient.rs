crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ReduceGradientOp<InputTypes,Context,Reducer> {
    storage:   OperatorStorage,
    context:   Context,
    axes:      Vec<i32>,// {};
    reducer:   Reducer,
    phantomIT: PhantomData<InputTypes>,
}

impl<InputTypes,Context,Reducer> ReduceGradientOp<InputTypes,Context,Reducer> {
    
    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            axes_(this->template GetRepeatedArgument<int>("axes"))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<InputTypes>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
    
        todo!();
        /*
            const auto& dY = Input(0);
        const auto& X = Input(1);
        const auto& Y = Input(2);

        const int ndim = X.dim();
        if (axes_.empty()) {
          axes_.resize(ndim);
          std::iota(axes_.begin(), axes_.end(), 0);
        } else {
          for (auto& axis : axes_) {
            axis = X.canonical_axis_index(axis);
          }
          std::sort(axes_.begin(), axes_.end());
          CAFFE_ENFORCE_GE(axes_.front(), 0, "Axes ids must be non-negative.");
          CAFFE_ENFORCE_LT(
              axes_.back(),
              ndim,
              "Axes ids must be smaller than the dimensions of input.");
        }
        const std::vector<int> dX_dims(X.sizes().cbegin(), X.sizes().cend());
        std::vector<int> dY_dims = dX_dims;
        for (const int axis : axes_) {
          dY_dims[axis] = 1;
        }
        auto* dX = Output(0, X.sizes(), at::dtype<T>());
        return reducer_.template Backward<T>(
            dY_dims,
            dX_dims,
            dY.template data<T>(),
            X.template data<T>(),
            Y.template data<T>(),
            dX->template mutable_data<T>(),
            &context_);
        */
    }
}
