crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SelectGradientOpBase<T,Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

impl<T, Context> SelectGradientOpBase<T, Context> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& Y = Input(0);
      const auto& dY = Input(1);
      const int N = Y.numel();
      ConstEigenVectorArrayMap<T> Y_arr(Y.template data<T>(), N);
      ConstEigenVectorArrayMap<T> dY_arr(dY.template data<T>(), N);
      for (int i = 0; i < OutputSize(); i++) {
        const auto& Xi = Input(i + 2);
        auto* dXi = Output(i, Xi.sizes(), at::dtype<T>());
        ConstEigenVectorArrayMap<T> Xi_arr(Xi.template data<T>(), N);
        EigenVectorArrayMap<T> dXi_arr(dXi->template mutable_data<T>(), N);
        dXi_arr = (Xi_arr == Y_arr).template cast<T>() * dY_arr;
      }
      return true;
        */
    }
}
