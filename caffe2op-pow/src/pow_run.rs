crate::ix!();

impl<InputTypes,Context,Functor,TypeMap> PowOp<InputTypes,Context,Functor,TypeMap> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<InputTypes>::call(this, Input(0));
        */
    }
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
    
        todo!();
        /*
            if ((InputSize() == 1) && HasArgument("exponent")) { // UnaryElementwiseOp
          const auto& A = Input(0);

          auto* C =
              Output(0, A.sizes(), at::dtype<typename TypeMap::template type<T>>());
          const T* Adata = A.template data<T>();
          auto* Cdata =
              C->template mutable_data<typename TypeMap::template type<T>>();
          functor_.template Run<true, T, float, T>(
              A.numel(), Adata, NULL, exponent_, Cdata, &context_);
        } else if (InputSize() == 2) { // BinaryElementwiseOp
          const auto& A = Input(0);
          const auto& B = Input(1);
          CAFFE_ENFORCE(
              !IsInputOutputAlias(1, 0) || !enable_broadcast_,
              "In-place is allowed only with the first tensor when broadcasting");
          auto* C =
              Output(0, A.sizes(), at::dtype<typename TypeMap::template type<T>>());
          const T* Adata = A.template data<T>();
          const T* Bdata = B.template data<T>();
          auto* Cdata =
              C->template mutable_data<typename TypeMap::template type<T>>();
          if (!enable_broadcast_) {
            CAFFE_ENFORCE_EQ(
                A.sizes(),
                B.sizes(),
                "Dimension mismatch - did you forget to set broadcast=1?");
            functor_.template Run<false, T, T, T>(
                A.numel(), Adata, Bdata, 0, Cdata, &context_);
          } else if (B.numel() == 1) {
            functor_.template Run<true, T, T, T>(
                A.numel(), Adata, Bdata, 0, Cdata, &context_);
          } else {
            size_t pre, n, post;
            std::tie(pre, n, post) =
                elementwise_ops_utils::ComputeLegacyBroadcastSizes(A, B, axis_);
            if (post == 1) {
              functor_.template RunWithBroadcast<T, T, T>(
                  Adata, Bdata, Cdata, pre, n, &context_);
            } else {
              functor_.template RunWithBroadcast2<T, T, T>(
                  Adata, Bdata, Cdata, pre, n, post, &context_);
            }
          }
        } else {
          CAFFE_THROW(
              "Only a tensor with an argument or two input tensors are supported as input to pow operator.");
        }
        return true;
        */
    }
}
