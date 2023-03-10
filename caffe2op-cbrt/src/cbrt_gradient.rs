crate::ix!();

pub struct CbrtGradientFunctor<Context> {
    phantom: PhantomData<Context>,
}

num_inputs!{CbrtGradient, 2}

num_outputs!{CbrtGradient, 1}

register_cpu_operator!{
    CbrtGradient,
    BinaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        CbrtGradientFunctor<CPUContext>>
}

allow_inplace!{CbrtGradient, vec![(0, 0)]}

identical_type_and_shape_of_input!{CbrtGradient, 0}

impl CbrtGradientFunctor<CPUContext> {

    #[inline] pub fn forward<T>(
        &self, 
        dY_dims:  &Vec<i32>,
        y_dims:   &Vec<i32>,
        dY:       *const T,
        y:        *const T,
        dX:       *mut T,
        context:  *mut CPUContext) -> bool 
    {
        todo!();
        /*
            const int size = std::accumulate(
              dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
          EigenVectorMap<T>(dX, size) = ConstEigenVectorArrayMap<T>(dY, size) /
              ConstEigenVectorArrayMap<T>(Y, size).square() / T(3);
          return true;
        */
    }
}
