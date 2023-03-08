crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MaxReductionOp<T,Context,const ROWWISE: bool> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

impl<T, Context, const ROWWISE: bool> MaxReductionOp<T,Context,ROWWISE> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
        CAFFE_ENFORCE_EQ(X.dim(), 3);

        const int batch_size = X.dim32(0);
        const int M = X.dim32(1);
        const int N = X.dim32(2);

        auto* Y = Output(0, {batch_size, ROWWISE ? M : N}, at::dtype<T>());

        if (ROWWISE) {
          math::RowwiseMax<T, Context>(
              batch_size * M,
              N,
              X.template data<T>(),
              Y->template mutable_data<T>(),
              &context_);
        } else {
          const int input_size = N * M;
          for (int i = 0; i < batch_size; ++i) {
            math::ColwiseMax<T, Context>(
                M,
                N,
                X.template data<T>() + i * input_size,
                Y->template mutable_data<T>() + i * N,
                &context_);
          }
        }
        return true;
        */
    }
}
