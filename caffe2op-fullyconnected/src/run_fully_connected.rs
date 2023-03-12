crate::ix!();

impl<Context, Engine, const TransposeWeight: bool> 
FullyConnectedOp<Context, Engine, TransposeWeight> {
    
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

    #[inline] pub fn do_run_with_type<T_X, T_W, T_B, T_Y, MATH>(&mut self) -> bool {
        todo!();
        /*
            const auto& X = Input(0);
            const auto& W = Input(1);
            const auto& b = Input(2);

            CAFFE_ENFORCE(b.dim() == 1, b.dim());
            // batch size
            const auto canonical_axis = X.canonical_axis_index(axis_);
            const auto M = X.size_to_dim(canonical_axis);
            const auto K = X.size_from_dim(canonical_axis);
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
                  ", b: ",
                  b.sizes(),
                  ", axis: ",
                  axis_,
                  ", M: ",
                  M,
                  ", N: ",
                  N,
                  ", K: ",
                  K);
            };

            // Error checking
            CAFFE_ENFORCE(M == X.numel() / K, dimErrorString());
            CAFFE_ENFORCE(K == W.numel() / N, dimErrorString());
            CAFFE_ENFORCE(N == b.dim32(0), dimErrorString());
            CAFFE_ENFORCE(N == b.numel(), dimErrorString());

            Y_shape_cache_ = X.sizes().vec();
            // This is an invariant of canonical_axis, so we can DCHECK.
            DCHECK_LE(canonical_axis + 1, Y_shape_cache_.size());
            Y_shape_cache_.resize(canonical_axis + 1);
            Y_shape_cache_[canonical_axis] = N;
            auto* Y = Output(0, Y_shape_cache_, at::dtype<T_Y>());
            CAFFE_ENFORCE(M * N == Y->numel(), dimErrorString());

            if (X.numel() == 0) {
              // skip the rest of the computation if X is empty
              Y->template mutable_data<T_Y>();
              return true;
            }

            // default to FLOAT as math.h does.
            TensorProto::DataType math_type = TensorProto_DataType_FLOAT;
            if (fp16_type<MATH>()) {
              math_type = TensorProto_DataType_FLOAT16;
            }

            // W * x
            math::Gemm<T_X, Context, Engine>(
                CblasNoTrans,
                TransposeWeight ? CblasTrans : CblasNoTrans,
                M,
                N,
                K,
                1,
                X.template data<T_X>(),
                W.template data<T_W>(),
                0,
                Y->template mutable_data<T_Y>(),
                &context_,
                math_type);

            // Add bias term
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

            math::Gemm<T_B, Context, Engine>(
                CblasNoTrans,
                CblasNoTrans,
                M,
                N,
                1,
                1,
                bias_multiplier_->template data<T_B>(),
                b.template data<T_B>(),
                1,
                Y->template mutable_data<T_Y>(),
                &context_,
                math_type);

            return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DoRunWithType<
            float, // X
            float, // W
            float, // B
            float, // Y
            float>(); // Math
        */
    }
}
