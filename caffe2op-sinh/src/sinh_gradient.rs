crate::ix!();

pub struct SinhGradientFunctor<Context> {
    phantom: PhantomData<Context>,
}

impl SinhGradientFunctor<CPUContext> {
    
    #[inline] pub fn forward<T>(&self, 
        dy_dims: &Vec<i32>,
        x_dims:  &Vec<i32>,
        dy:      *const T,
        x:       *const T,
        dx:      *mut T,
        context: *mut CPUContext) -> bool {

        todo!();
        /*
            const int size = std::accumulate(
          X_dims.cbegin(), X_dims.cend(), 1, std::multiplies<int>());
      ConstEigenVectorArrayMap<T> dY_arr(dY, size);
      ConstEigenVectorArrayMap<T> X_arr(X, size);
      EigenVectorMap<T>(dX, size) = dY_arr * (X_arr.exp() + (-X_arr).exp()) / 2;
      return true;
        */
    }
}

register_cpu_operator!{SinhGradient,
    BinaryElementwiseOp<
        TensorTypes<float>,
        CPUContext,
        SinhGradientFunctor<CPUContext>>}

num_inputs!{SinhGradient, 2}

num_outputs!{SinhGradient, 1}

identical_type_and_shape_of_input!{SinhGradient, 0}
