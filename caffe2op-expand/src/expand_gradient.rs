crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ExpandGradientOp<InputTypes, Context> {
    storage:   OperatorStorage,
    context:   Context,
    phantomIT: PhantomData<InputTypes>,
}

num_inputs!{ExpandGradient, 2}

num_outputs!{ExpandGradient, 1}

register_cpu_operator!{
    ExpandGradient,
    ExpandGradientOp<
        TensorTypes<i32, i64, f32, f64>,
        CPUContext>
}

register_cuda_operator!{
    ExpandGradient,
    ExpandGradientOp<
        TensorTypes<i32, i64, f32, f64>,
        CUDAContext>
}

impl<InputTypes,Context> ExpandGradientOp<InputTypes,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
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

            const int ndim = dY.dim();
            const std::vector<int> dX_dims(X.sizes().cbegin(), X.sizes().cend());
            const std::vector<int> dY_dims(dY.sizes().cbegin(), dY.sizes().cend());
            auto* dX = Output(0, X.sizes(), at::dtype<T>());
            std::vector<int> axes;
            const int offset = ndim - X.dim();
            for (int i = 0; i < ndim; i++) {
              if (i < offset || dX_dims[i - offset] == 1) {
                axes.push_back(i);
              }
            }
            std::vector<int> X_dims = dY_dims;
            for (const int axis : axes) {
              X_dims[axis] = 1;
            }
            math::ReduceSum<T, Context>(
                dY_dims.size(),
                dY_dims.data(),
                X_dims.data(),
                T(1),
                dY.template data<T>(),
                dX->template mutable_data<T>(),
                &context_);
            return true;
        */
    }
}
