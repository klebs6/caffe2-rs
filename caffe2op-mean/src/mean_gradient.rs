crate::ix!();

///-----------------------------------------
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MeanGradientOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

num_inputs!{MeanGradient, 1}

num_outputs!{MeanGradient, (1,INT_MAX)}

allow_inplace!{MeanGradient, vec![(0, 0)]}

impl<Context> MeanGradientOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& dY = Input(0);
        const auto* dY_data = dY.template data<T>();
        int size = dY.numel();

        int num_inputs = OutputSize();
        float scale = 1.0f / num_inputs;

        // dX0 = scale * dY

        auto* dX0 = Output(0, dY.sizes(), at::dtype<T>());
        math::Scale(
            size, scale, dY_data, dX0->template mutable_data<T>(), &context_);

        // Copy the rest dX
        for (int i = 1; i < num_inputs; i++) {
          auto* cur_dX = Output(i);
          cur_dX->ResizeLike(dY);
          cur_dX->CopyFrom(*dX0, true /*async*/);
        }

        return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (Input(0).template IsType<float>()) {
          return DoRunWithType<float>();
        } else if (Input(0).template IsType<double>()) {
          return DoRunWithType<double>();
        } else {
          CAFFE_THROW(
              "Mean operator only supports 32-bit float or 64-bit double, but",
              " input was of type ",
              Input(0).dtype().name());
        }
        */
    }
}
