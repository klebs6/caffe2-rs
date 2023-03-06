crate::ix!();

/**
  | Mish takes one input data (Tensor) and
  | produces one output data (Tensor) where
  | the Mish function, y = x * tanh(ln(1 +
  | exp(x))), is applied to the tensor elementwise.
  |
  */
pub struct MishFunctor<Context> { 
    // Input: X, output: Y
    phantom: PhantomData<Context>,
}

num_inputs!{Mish, 1}

num_outputs!{Mish, 1}

inputs!{Mish, 
    0 => ("X", "1D input tensor")
}

outputs!{Mish, 
    0 => ("Y", "1D output tensor")
}

identical_type_and_shape!{Mish}

register_cpu_operator!{
    Mish,
    UnaryElementwiseOp<
        TensorTypes<float, double>,
        CPUContext,
        MishFunctor<CPUContext>>
}

impl MishFunctor<CPUContext> {

    #[inline] pub fn invoke<T>(&self, 
        n:       i32,
        x:       *const T,
        y:       *mut T,
        context: *mut CPUContext) -> bool {
        todo!();
        /*
          ConstEigenVectorArrayMap<T> X_arr(X, N);
          EigenVectorArrayMap<T> Y_arr(Y, N);
          math::Exp<T, CPUContext>(N, X, Y, context);
          math::Log1p<T, CPUContext>(N, Y, Y, context);
          Y_arr = X_arr * Y_arr.tanh();
          return true;
        */
    }
}
