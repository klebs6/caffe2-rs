crate::ix!();

impl<Context, Engine,  const TransposeWeight: bool> 
FullyConnectedGradientOp<Context, Engine, TransposeWeight> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            axis_(this->template GetSingleArgument<int32_t>("axis", 1)),
            axis_w_(this->template GetSingleArgument<int32_t>("axis_w", 1)),
            float16_compute_(
                this->template GetSingleArgument<bool>("float16_compute", false))
        */
    }

    #[inline] pub fn do_run_with_type<T_X, T_W, T_DY, T_B, T_DX, T_DW, T_DB, MATH>(&mut self) -> bool {
        todo!();
        /*
            const auto& X = Input(0);
            const auto& W = Input(1);
            const auto& dY = Input(2);
            // batch size
            const auto canonical_axis = X.canonical_axis_index(axis_);
            const int M = X.size_to_dim(canonical_axis);
            const int K = X.size_from_dim(canonical_axis);
            const auto canonical_axis_w = W.canonical_axis_index(axis_w_);
            const int N = TransposeWeight ? W.size_to_dim(canonical_axis_w)
                                          : W.size_from_dim(canonical_axis_w);

            auto dimErrorString = [&]() {
              return c10::str(
                  "Dimension mismatch: ",
                  "X: ",
                  X.sizes(),
                  ", W: ",
                  W.sizes(),
                  ", dY: ",
                  dY.sizes(),
                  ", axis: ",
                  axis_,
                  ", M: ",
                  M,
                  ", N: ",
                  N,
                  ", K: ",
                  K);
            };

            CAFFE_ENFORCE(M * K == X.numel(), dimErrorString());
            CAFFE_ENFORCE(K * N == W.numel(), dimErrorString());

            auto* dW = Output(0, W.sizes(), at::dtype<T_DW>());
            auto* db = Output(1, {N}, at::dtype<T_DB>());

            if (X.numel() == 0) {
              // generate a zero blob for db and dW when X is empty
              math::Set<T_DB, Context>(
                  db->numel(),
                  convert::To<float, T_DB>(0),
                  db->template mutable_data<T_DB>(),
                  &context_);
              math::Set<T_DW, Context>(
                  dW->numel(),
                  convert::To<float, T_DW>(0),
                  dW->template mutable_data<T_DW>(),
                  &context_);

              if (OutputSize() == 3) {
                Output(2, X.sizes(), at::dtype<T_DX>());
              }

              return true;
            }

            // default to FLOAT as math.h does.
            TensorProto::DataType math_type = TensorProto_DataType_FLOAT;
            if (fp16_type<MATH>()) {
              math_type = TensorProto_DataType_FLOAT16;
            }

            // Compute dW
            math::Gemm<T_DY, Context, Engine>(
                CblasTrans,
                CblasNoTrans,
                TransposeWeight ? N : K,
                TransposeWeight ? K : N,
                M,
                1,
                TransposeWeight ? dY.template data<T_DY>() : X.template data<T_X>(),
                TransposeWeight ? X.template data<T_X>() : dY.template data<T_DY>(),
                0,
                dW->template mutable_data<T_DW>(),
                &context_,
                math_type);
            if (!bias_multiplier_.has_value()) {
              bias_multiplier_ =
                  caffe2::empty({M}, at::dtype<T_B>().device(Context::GetDeviceType()));
              math::Set<T_B, Context>(
                  M,
                  convert::To<float, T_B>(1),
                  bias_multiplier_->template mutable_data<T_B>(),
                  &context_);
            } else if (bias_multiplier_->numel() != M) {
              bias_multiplier_->Resize(M);
              math::Set<T_B, Context>(
                  M,
                  convert::To<float, T_B>(1),
                  bias_multiplier_->template mutable_data<T_B>(),
                  &context_);
            }
            // Compute dB
            math::Gemv<T_DY, Context>(
                CblasTrans,
                M,
                N,
                1,
                dY.template data<T_DY>(),
                bias_multiplier_->template data<T_B>(),
                0,
                db->template mutable_data<T_DB>(),
                &context_);

            // Compute dX
            if (OutputSize() == 3) {
              auto* dX = Output(2, X.sizes(), at::dtype<T_DX>());
              math::Gemm<T_DX, Context, Engine>(
                  CblasNoTrans,
                  TransposeWeight ? CblasNoTrans : CblasTrans,
                  M,
                  K,
                  N,
                  1,
                  dY.template data<T_DY>(),
                  W.template data<T_W>(),
                  0,
                  dX->template mutable_data<T_DX>(),
                  &context_,
                  math_type);
            }
            return true;
        */
    }

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DoRunWithType<
            float, //  X
            float, //  W
            float, // dY
            float, //  B
            float, // dX
            float, // dW
            float, // dB
            float>(); // Math
        */
    }
}
