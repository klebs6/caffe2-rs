crate::ix!();

impl HardSigmoidFunctor<CPUContext> {

    #[inline] pub fn invoke<T>(
        &self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut CPUContext) -> bool 
    {
        todo!();
        /*
            EigenVectorArrayMap<T>(Y, N) =
              (ConstEigenVectorArrayMap<T>(X, N) * T(alpha) + T(beta))
                  .cwiseMin(T(1))
                  .cwiseMax(T(0));
          return true;
        */
    }
}

register_cpu_operator!{
    HardSigmoid,
    UnaryElementwiseWithArgsOp<
        TensorTypes<f32>,
        CPUContext,
        HardSigmoidFunctor<CPUContext>>
}

register_cpu_operator!{
    HardSigmoidGradient,
    BinaryElementwiseWithArgsOp<
        TensorTypes<f32>,
        CPUContext,
        HardSigmoidGradientFunctor<CPUContext>>
}

