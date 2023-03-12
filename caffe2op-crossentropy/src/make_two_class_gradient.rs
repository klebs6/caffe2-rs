crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MakeTwoClassGradientOp<T, Context> {

    storage: OperatorStorage,
    context: Context,

    // Input: dY
    // Ouptut: dX
    phantom: PhantomData<T>,
}

num_inputs!{MakeTwoClassGradient, 1}

num_outputs!{MakeTwoClassGradient, 1}

impl MakeTwoClassGradientOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& dY = Input(0);

      auto shape = dY.sizes().vec();
      CAFFE_ENFORCE_GE(shape.size(), 1);
      CAFFE_ENFORCE_EQ(shape.back(), 2);
      shape.pop_back();
      auto* dX = Output(0, shape, at::dtype<float>());
      const float* dYdata = dY.data<float>();
      float* dXdata = dX->template mutable_data<float>();
      int64_t N = dX->numel();
      // use eigen?
      for (int64_t i = 0; i < N; ++i) {
        dXdata[i] = dYdata[i * 2 + 1] - dYdata[i * 2];
      }
      return true;
        */
    }
}
