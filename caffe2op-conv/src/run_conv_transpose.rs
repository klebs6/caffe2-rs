crate::ix!();

impl<T, Context> ConvTransposeOp<T, Context> {

    #[inline] pub fn run_on_device_with_order_nchw(&mut self) -> bool {
        todo!();
        /*
            const auto& X = Input(INPUT);
          const auto& filter = Input(FILTER);
          CAFFE_ENFORCE_EQ(X.dim(), 4, "Input must be 4D tensor");
          CAFFE_ENFORCE_EQ(filter.dim(), 4, "filter must be 4D tensor");
          const int N = X.dim32(0);
          const int M = X.dim32(1);
          const int H = X.dim32(2);
          const int W = X.dim32(3);
          const int G = group_;
          CAFFE_ENFORCE_EQ(M, filter.dim32(0));
          CAFFE_ENFORCE_EQ(
              M % G, 0, "The number of input channels is not divisible by group.");
          const int C = filter.dim32(1) * G;
          CAFFE_ENFORCE_EQ(
              filter.dim32(2),
              kernel_h(),
              "filter height must be equal to kernel height");
          CAFFE_ENFORCE_EQ(
              filter.dim32(3),
              this->kernel_w(),
              "filter width must be equal to kernel width");
          const std::vector<std::int64_t> Y_dims =
              ConvTransposeUnpoolBase<Context>::GetOutputSize(X, C);
          auto* Y = Output(0, Y_dims, at::dtype<T>());
          if (X.numel() == 0) {
            VLOG(2) << "Number of elements is 0 in ConvTrasposeOp";
            return true;
          }

          const int K_HxW = kernel_h() * kernel_w();
          const int kernel_dim = C / G * K_HxW;
          const int X_HxW = H * W;
          const int Y_HxW = Y->dim32(2) * Y->dim32(3);

          const T* X_data = X.template data<T>();
          const T* filter_data = filter.template data<T>();
          const T* bias_data = nullptr;
          if (InputSize() == 3) {
            auto& bias = Input(BIAS);
            CAFFE_ENFORCE_EQ(bias.dim(), 1, "bias must be 1D tensor");
            CAFFE_ENFORCE_EQ(
                bias.dim32(0),
                C,
                "bias dimension must be equal to output channel number");
            bias_data = bias.template data<T>();
          }
          T* Y_data = Y->template mutable_data<T>();

          const std::vector<std::int64_t> buffer_shape = {
              C, kernel_h(), kernel_w(), H, W};

          const auto func = [&](Tensor* col_buffer) {
            ReinitializeTensor(
                col_buffer,
                buffer_shape,
                at::dtype<T>().device(Context::GetDeviceType()));
            T* col_buffer_data = col_buffer->template mutable_data<T>();
            for (int image_id = 0; image_id < N; ++image_id) {
              // Weight term
              if (G == 1) {
                math::Gemm<T, Context>(
                    CblasTrans,
                    CblasNoTrans,
                    kernel_dim,
                    X_HxW,
                    M,
                    1.0f,
                    filter_data,
                    X_data + image_id * M * X_HxW,
                    0.0f,
                    col_buffer_data,
                    &context_);
              } else {
                math::GemmStridedBatched<T, Context>(
                    CblasTrans,
                    CblasNoTrans,
                    G,
                    kernel_dim,
                    X_HxW,
                    M / G,
                    1.0f,
                    filter_data,
                    M / G * kernel_dim,
                    X_data + image_id * M * X_HxW,
                    M / G * X_HxW,
                    0.0f,
                    col_buffer_data,
                    col_buffer->numel() / G,
                    &context_);
              }

              // Col2Im
              math::Col2Im<T, Context, StorageOrder::NCHW>(
                  C,
                  Y->dim32(2),
                  Y->dim32(3),
                  kernel_h(),
                  kernel_w(),
                  1,
                  1,
                  pad_t(),
                  pad_l(),
                  pad_b(),
                  pad_r(),
                  stride_h(),
                  stride_w(),
                  col_buffer_data,
                  Y_data + image_id * C * Y_HxW,
                  &context_);

              if (bias_data != nullptr) {
                // Bias term
        #if defined(__ARM_NEON__) || defined(__ARM_NEON)
                math::BiasCHW<T, Context>(
                    bias_data,
                    nullptr,
                    C,
                    Y_HxW,
                    Y_data + image_id * C * Y_HxW,
                    &context_);
        #endif // !defined(__ARM_NEON__) && !defined(__ARM_NEON)
              }
            }
            if (bias_data != nullptr) {
        #if !defined(__ARM_NEON__) && !defined(__ARM_NEON)
              // Bias term
              const std::array<int, 3> Y_dims = {N, C, Y_HxW};
              const std::array<int, 3> b_dims = {1, C, 1};
              math::Add<T, Context>(
                  3,
                  Y_dims.data(),
                  3,
                  b_dims.data(),
                  Y_data,
                  bias_data,
                  Y_data,
                  &context_);
        #endif
            }
          };

          if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
            runWithSharedBuffer<Context>(ws_, func);
          } else {
            func(&col_buffer_);
          }
          return true;
        */
    }

    #[inline] pub fn run_on_device_with_order_nhwc(&mut self) -> bool {
        todo!();
        /*
            const auto& X = Input(INPUT);
          auto& filter = Input(FILTER);
          CAFFE_ENFORCE_EQ(filter.dim(), 4, "filter must be 4D tensor");
          const int N = X.dim32(0);
          const int H = X.dim32(1);
          const int W = X.dim32(2);
          const int M = X.dim32(3);
          const int G = group_;
          CAFFE_ENFORCE_EQ(
              filter.dim32(0),
              M,
              "filter number must be equal to input channel number");
          CAFFE_ENFORCE_EQ(
              M % G, 0, "The number of input channels is not divisible by group.");
          const int C = filter.dim32(3) * G;
          CAFFE_ENFORCE_EQ(
              filter.dim32(1),
              kernel_h(),
              "filter height must be equal to kernel height");
          CAFFE_ENFORCE_EQ(
              filter.dim32(2),
              kernel_w(),
              "filter width must be equal to kernel width");

          const std::vector<std::int64_t> Y_dims =
              ConvTransposeUnpoolBase<Context>::GetOutputSize(X, C);
          auto* Y = Output(0, Y_dims, at::dtype<T>());
          if (X.numel() == 0) {
            VLOG(2) << "Number of elements is 0 in ConvTrasposeOp";
            return true;
          }

          const int K_HxW = kernel_h() * kernel_w();
          const int kernel_dim = C / G * K_HxW;
          const int X_HxW = H * W;
          const int Y_HxW = Y->dim32(1) * Y->dim32(2);

          const T* X_data = X.template data<T>();
          const T* filter_data = filter.template data<T>();
          const T* bias_data = nullptr;
          if (InputSize() == 3) {
            auto& bias = Input(BIAS);
            CAFFE_ENFORCE_EQ(bias.dim(), 1, "bias must be 1D tensor");
            CAFFE_ENFORCE_EQ(
                bias.dim32(0),
                C,
                "bias dimension must be equal to output channel number");
            bias_data = bias.template data<T>();
          }
          T* Y_data = Y->template mutable_data<T>();

          const std::vector<std::int64_t> buffer_shape = {
              G, H, W, kernel_h(), kernel_w(), C / G};
          const auto func = [&](Tensor* /*col_buffer*/) {
            ReinitializeTensor(
                &col_buffer_,
                buffer_shape,
                at::dtype<T>().device(Context::GetDeviceType()));
            T* col_buffer_data = col_buffer_.template mutable_data<T>();
            for (int image_id = 0; image_id < N; ++image_id) {
              // Weight term
              if (G == 1) {
                math::Gemm<T, Context>(
                    CblasNoTrans,
                    CblasNoTrans,
                    X_HxW,
                    kernel_dim,
                    M,
                    1.0f,
                    X_data + image_id * M * X_HxW,
                    filter_data,
                    0.0f,
                    col_buffer_data,
                    &context_);
              } else {
                for (int group_id = 0; group_id < G; ++group_id) {
                  math::GemmEx<T, Context>(
                      CblasNoTrans,
                      CblasNoTrans,
                      X_HxW,
                      kernel_dim,
                      M / G,
                      1.0f,
                      X_data + image_id * M * X_HxW + group_id * M / G,
                      M,
                      filter_data + group_id * M / G * kernel_dim,
                      kernel_dim,
                      0.0f,
                      col_buffer_data + group_id * kernel_dim,
                      G * kernel_dim,
                      &context_);
                }
              }
              // Col2Im
              math::Col2Im<T, Context, StorageOrder::NHWC>(
                  C,
                  Y->dim32(1),
                  Y->dim32(2),
                  kernel_h(),
                  kernel_w(),
                  1,
                  1,
                  pad_t(),
                  pad_l(),
                  pad_b(),
                  pad_r(),
                  stride_h(),
                  stride_w(),
                  col_buffer_data,
                  Y_data + image_id * C * Y_HxW,
                  &context_,
                  G);
            }
            if (bias_data != nullptr) {
              // Bias term
              const std::array<int, 2> Y_dims = {N * Y_HxW, C};
              const std::array<int, 2> b_dims = {1, C};
              math::Add<T, Context>(
                  2,
                  Y_dims.data(),
                  2,
                  b_dims.data(),
                  Y_data,
                  bias_data,
                  Y_data,
                  &context_);
            }
          };

          if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
            runWithSharedBuffer<Context>(ws_, func);
          } else {
            func(&col_buffer_);
          }
          return true;
        */
    }
}

