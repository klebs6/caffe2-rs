crate::ix!();

impl SumReduceLikeOp<CPUContext> {

    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            const auto& A = Input(0);
          const auto& B = Input(1);

          CAFFE_ENFORCE(!IsInputOutputAlias(1, 0), "In-place is not allowed.");
          auto* C = Output(0, B.sizes(), at::dtype<T>());
          const T* Adata = A.template data<T>();
          auto* Cdata = C->template mutable_data<T>();
          if (B.numel() == 1) {
            auto count = A.numel();
            SRLHelper::sum2one<T>(Adata, Cdata, count);
          } else {
            size_t pre, n, post;
            std::tie(pre, n, post) =
                elementwise_ops_utils::ComputeLegacyBroadcastSizes(A, B, axis_);
            if (post == 1) {
              SRLHelper::RunWithBroadcastFront<T>(Adata, Cdata, pre, n, &context_);
            } else if (pre == 1) {
              SRLHelper::RunWithBroadcastBack<T>(Adata, Cdata, post, n, &context_);
            } else {
              SRLHelper::RunWithBroadcast2<T>(Adata, Cdata, pre, n, post, &context_);
            }
          }
          return true;
        */
    }
}

register_cpu_operator!{
    SumReduceLike, 
    SumReduceLikeOp<CPUContext>
}
