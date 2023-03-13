crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct WeightedSumGradientOp<Context> {
    storage:   OperatorStorage,
    context:   Context,
    grad_on_w: bool,
}

num_outputs!{WeightedSumGradient, (1,INT_MAX)}

num_inputs!{WeightedSumGradient, 
    |n: i32| {
        n > 0 && n % 2 == 1
    }
}

impl<Context> WeightedSumGradientOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            grad_on_w_(this->template GetSingleArgument<bool>("grad_on_w", false))
        */
    }
    
    #[inline] pub fn do_run_with_type<DstType>(&mut self) -> bool {
        todo!();
        /*
            CAFFE_ENFORCE_EQ(InputSize() % 2, 1);
        auto output_size = grad_on_w_ ? InputSize() - 1 : InputSize() / 2;
        CAFFE_ENFORCE_EQ(OutputSize(), output_size);

        auto& dY = Input(0);
        const auto* dY_data = dY.template data<DstType>();
        int size = dY.numel();

        // The input size should be the input size of the forward op plus 1
        for (int i = 0; i < InputSize() / 2; i++) {
          auto& cur_w = Input(2 * i + 2);
          CAFFE_ENFORCE_EQ(cur_w.numel(), 1);

          auto* cur_dX = Output(i, dY.sizes(), at::dtype<DstType>());

          math::Scale<float, DstType, Context>(
              size,
              cur_w.template data<float>(),
              dY_data,
              cur_dX->template mutable_data<DstType>(),
              &context_);

          if (grad_on_w_) {
            auto& cur_X = Input(2 * i + 1);
            CAFFE_ENFORCE_EQ(cur_X.numel(), size);
            auto* cur_dw = Output(i + output_size / 2);
            cur_dw->Resize(1);
            math::Dot<DstType, Context>(
                size,
                dY_data,
                cur_X.template data<DstType>(),
                cur_dw->template mutable_data<float>(),
                &context_);
          }
        }

        return true;
        */
    }
}
