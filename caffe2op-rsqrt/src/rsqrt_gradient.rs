crate::ix!();

pub struct RsqrtGradientFunctor<Context> {
    
    phantom: PhantomData<Context>,
}

num_inputs!{RsqrtGradient, 2}

num_outputs!{RsqrtGradient, 1}

allow_inplace!{RsqrtGradient, vec![(0, 0)]}

impl RsqrtGradientFunctor<CPUContext> {

    #[inline] pub fn forward<T>(&self, 
        dy_dims: &Vec<i32>,
        y_dims:  &Vec<i32>,
        dy:      *const T,
        y:       *const T,
        dx:      *mut T,
        context: *mut CPUContext) -> bool {
    
        todo!();
        /*
            const int size = std::accumulate(
          dY_dims.cbegin(), dY_dims.cend(), 1, std::multiplies<int>());
      EigenVectorMap<T>(dX, size) = ConstEigenVectorMap<T>(dY, size).array() *
          ConstEigenVectorMap<T>(Y, size).array().cube() * static_cast<T>(-0.5);
      return true;
        */
    }
}
