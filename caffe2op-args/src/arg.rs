crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ArgOp<Context, Reducer> {
    storage:   OperatorStorage,
    context:   Context,
    axis:      i32,
    reducer:   Reducer,
    keep_dims: bool,
}

should_not_do_gradient!{ArgMax}
should_not_do_gradient!{ArgMin}

impl<Context,Reducer> ArgOp<Context, Reducer> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "axis", axis_, -1),
            OP_SINGLE_ARG(bool, "keepdims", keep_dims_, true)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<
            TensorTypes<std::int32_t, std::int64_t, float, double>>::
            call(this, Input(0));
        */
    }

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto& X = Input(0);

            const int ndim = X.dim();
            if (axis_ == -1) {
              axis_ = ndim - 1;
            }
            CAFFE_ENFORCE_GE(axis_, 0);
            CAFFE_ENFORCE_LT(axis_, ndim);
            const std::vector<int> X_dims(X.sizes().cbegin(), X.sizes().cend());
            std::vector<int64_t> Y_dims;
            Y_dims.reserve(ndim);
            int prev_size = 1;
            int next_size = 1;
            for (int i = 0; i < axis_; ++i) {
              Y_dims.push_back(X_dims[i]);
              prev_size *= X_dims[i];
            }
            if (keep_dims_) {
              Y_dims.push_back(1);
            }
            for (int i = axis_ + 1; i < ndim; ++i) {
              Y_dims.push_back(X_dims[i]);
              next_size *= X_dims[i];
            }
            auto* Y = Output(0, Y_dims, at::dtype<int64_t>());
            const int n = X_dims[axis_];
            return reducer_(
                prev_size,
                next_size,
                n,
                X.template data<T>(),
                Y->template mutable_data<int64_t>(),
                &context_);
        */
    }
}
