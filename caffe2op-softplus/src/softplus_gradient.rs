crate::ix!();

pub struct SoftplusGradientOp<DataType> {

    storage:         OperatorStorage,
    context:         CPUContext,

    /**
      | Input: Y, dY,
      | 
      | output: dX
      |
      */
    phantomDataType: PhantomData<DataType>,
}

num_inputs!{SoftplusGradient, 2}

num_outputs!{SoftplusGradient, 1}

allow_inplace!{SoftplusGradient, vec![(1, 0)]}

register_cpu_operator!{
    Softplus,         
    SoftplusOp<f32, CPUContext>
}

register_cpu_operator!{
    SoftplusGradient, 
    SoftplusGradientOp<f32, CPUContext>
}

impl<DataType> SoftplusGradientOp<DataType> {

    fn run_on_device() -> bool {
        todo!();
        /*
          auto& Y = Input(0);
          auto& dY = Input(1);

          DCHECK_EQ(dY.numel(), Y.numel());
          auto* dX = Output(0, Y.sizes(), at::dtype<float>());

          const float* Ydata = Y.data<float>();
          const float* dYdata = dY.data<float>();
          float* dXdata = dX->template mutable_data<float>();
          EigenVectorArrayMap<float> dXvec(dXdata, dX->numel());
          ConstEigenVectorArrayMap<float> Yvec(Ydata, Y.numel());
          ConstEigenVectorArrayMap<float> dYvec(dYdata, dY.numel());
          dXvec = dYvec * (1.0 - (-Yvec).exp());
          return true;
        */
    }
}
