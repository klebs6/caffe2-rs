crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ReduceOp<InputTypes,Context,Reducer> {
    storage:   OperatorStorage,
    context:   Context,
    axes:      Vec<i32>,
    keep_dims: i32, // {};
    reducer:   Reducer,
    phantomIT: PhantomData<InputTypes>,
}

impl<InputTypes,Context,Reducer> ReduceOp<InputTypes,Context,Reducer> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            axes_(this->template GetRepeatedArgument<int>("axes")),
            OP_SINGLE_ARG(bool, "keepdims", keep_dims_, true)
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
            const auto& X = Input(0);
        const int ndim = X.dim();
        const std::vector<int> X_dims(X.sizes().cbegin(), X.sizes().cend());
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
        std::vector<int64_t> output_dims;
        output_dims.reserve(ndim);
        std::size_t cur_axis = 0;
        for (int i = 0; i < ndim; ++i) {
          if (cur_axis < axes_.size() && i == axes_[cur_axis]) {
            if (keep_dims_) {
              output_dims.push_back(1);
            }
            ++cur_axis;
          } else {
            output_dims.push_back(X_dims[i]);
          }
        }
        auto* Y = Output(0, output_dims, at::dtype<T>());

        std::vector<int> Y_dims = X_dims;
        for (const int axis : axes_) {
          Y_dims[axis] = 1;
        }

        return reducer_.template Forward<T>(
            X_dims,
            Y_dims,
            X.template data<T>(),
            Y->template mutable_data<T>(),
            &context_);
        */
    }
}
