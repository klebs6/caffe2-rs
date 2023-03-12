crate::ix!();

impl<T,Context> ConvGradientOp<T, Context> {

    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        todo!();
        /*
            auto& X = Input(INPUT);
          auto& filter = Input(FILTER);
          auto& dY = Input(OUTPUT_GRAD);

          const int N = X.dim32(0), C = X.dim32(1);

          const vector<int> input_dims = this->GetDims(X);
          const int input_image_size = this->GetDimsSize(X);

          const vector<int> output_dims = this->GetDims(dY);
          // The output image size is the spatial size of the output.
          const int output_image_size = this->GetDimsSize(dY);

          ConvPoolOpBase<Context>::ComputePads(input_dims);
          CAFFE_ENFORCE_EQ(X.dim(), filter.dim());
          const int M = filter.dim32(0);
          CAFFE_ENFORCE_EQ(C, filter.dim32(1) * group_);

          int kernel_dims_size = 1;
          for (int i = 0; i < kernel_.size(); ++i) {
            CAFFE_ENFORCE_EQ(filter.dim32(i + 2), kernel_[i]);
            kernel_dims_size *= kernel_[i];
          }

          CAFFE_ENFORCE_EQ(M % group_, 0);
          auto* dfilter = Output(FILTER_GRAD, filter.sizes(), at::dtype<T>());
          // The dimension of each kernel
          const int kernel_dim = C / group_ * kernel_dims_size;
          // The col buffer is stored in CHW order as well - kernel_dim, and the height
          // and width.
          vector<int> img_shape;
          img_shape.assign(X.sizes().begin() + 1, X.sizes().end());
          vector<int> col_buffer_shape;
          col_buffer_shape.push_back(C / group_ * kernel_dims_size);
          col_buffer_shape.insert(
              col_buffer_shape.end(), output_dims.begin(), output_dims.end());
          vector<int64_t> col_buffer_shape_64;
          std::copy(
              col_buffer_shape.cbegin(),
              col_buffer_shape.cend(),
              std::back_inserter(col_buffer_shape_64));
          ReinitializeTensor(
              &col_buffer_,
              col_buffer_shape_64,
              at::dtype<T>().device(Context::GetDeviceType()));

          if (kernel_.size() != 2) {
            // TODO: SetDeviceTensor accept vector<int64_t>
            SetDeviceTensor(img_shape, &img_shape_device_);
            SetDeviceTensor(col_buffer_shape, &col_buffer_shape_device_);
          }

          const int col_buffer_size =
              (C / group_) * kernel_dims_size * output_image_size;
          const T* Xdata = X.template data<T>();
          const T* filter_data = filter.template data<T>();
          const T* dYdata = dY.template data<T>();
          T* col_buffer_data = col_buffer_.template mutable_data<T>();
          T* dfilter_data = dfilter->template mutable_data<T>();

          // Pre-setting the gradients to zero.
          math::Set<T, Context>(dfilter->numel(), 0, dfilter_data, &context_);

          T* dbias_data = nullptr;
          if (!no_bias_) {
            auto* dbias = Output(BIAS_OR_INPUT_GRAD, {M}, at::dtype<T>());
            // Removed the check for whether bias_multiplier_ has correct size or not
            ReinitializeTensor(
                &bias_multiplier_,
                vector<int64_t>(1, output_image_size),
                at::dtype<T>().device(Context::GetDeviceType()));
            math::Set<T, Context>(
                output_image_size,
                static_cast<T>(1),
                bias_multiplier_.template mutable_data<T>(),
                &context_);
            dbias_data = dbias->template mutable_data<T>();
            math::Set<T, Context>(dbias->numel(), 0, dbias_data, &context_);
          }

          if (N == 0) {
            if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
              auto* dX = Output(
                  no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD,
                  X.sizes(),
                  at::dtype<T>());
              dX->template mutable_data<T>();
            }
            return true;
          }

          // The offset corresponding to a single input image, and a single output
          // image.
          const int input_offset = C / group_ * input_image_size;
          const int output_offset = dY.numel() / dY.dim32(0) / group_;
          const int filter_offset = filter.numel() / group_;
          for (int image_id = 0; image_id < N; ++image_id) {
            for (int group_id = 0; group_id < group_; ++group_id) {
              // When we compute the gradient with respect to the filters, we need to do
              // im2col to allow gemm-type computation.
              if (kernel_.size() == 2) {
                math::Im2Col<T, Context, StorageOrder::NCHW>(
                    C / group_,
                    input_dims[0],
                    input_dims[1],
                    kernel_h(),
                    kernel_w(),
                    dilation_h(),
                    dilation_w(),
                    pad_t(),
                    pad_l(),
                    pad_b(),
                    pad_r(),
                    stride_h(),
                    stride_w(),
                    Xdata + group_id * input_offset,
                    col_buffer_data,
                    &context_);
              } else {
                math::Im2ColNd<T, Context, StorageOrder::NCHW>(
                    kernel_.size(),
                    input_offset,
                    col_buffer_size,
                    img_shape.data(),
                    col_buffer_shape.data(),
                    kernel_.data(),
                    stride_.data(),
                    dilation_.data(),
                    pads_.data(),
                    Xdata + group_id * input_offset,
                    col_buffer_data,
                    &context_);
              }
              // Gradient with respect to filter.
              math::Gemm<T, Context>(
                  CblasNoTrans,
                  CblasTrans,
                  M / group_,
                  kernel_dim,
                  output_image_size,
                  1,
                  dYdata + group_id * output_offset,
                  col_buffer_data,
                  1,
                  dfilter_data + group_id * filter_offset,
                  &context_);
            }
            if (!no_bias_) {
              // Gradient with respect to bias can be computed independent from group.
              math::Gemv<T, Context>(
                  CblasNoTrans,
                  M,
                  output_image_size,
                  1,
                  dYdata,
                  bias_multiplier_.template data<T>(),
                  1,
                  dbias_data,
                  &context_);
            }
            Xdata += input_offset * group_;
            dYdata += output_offset * group_;
          }
          if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
            // Compute the gradient w.r.t. the input.

            auto* dX = Output(
                no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD, X.sizes(), at::dtype<T>());
            T* dXdata = dX->template mutable_data<T>();
            dYdata = dY.template data<T>();
            for (int image_id = 0; image_id < N; ++image_id) {
              for (int group_id = 0; group_id < group_; ++group_id) {
                // Compute gradient into col_buffer.
                math::Gemm<T, Context>(
                    CblasTrans,
                    CblasNoTrans,
                    kernel_dim,
                    output_image_size,
                    M / group_,
                    1,
                    filter_data + group_id * filter_offset,
                    dYdata,
                    0,
                    col_buffer_data,
                    &context_);
                if (kernel_.size() == 2) {
                  math::Col2Im<T, Context, StorageOrder::NCHW>(
                      C / group_,
                      input_dims[0],
                      input_dims[1],
                      kernel_h(),
                      kernel_w(),
                      dilation_h(),
                      dilation_w(),
                      pad_t(),
                      pad_l(),
                      pad_b(),
                      pad_r(),
                      stride_h(),
                      stride_w(),
                      col_buffer_data,
                      dXdata,
                      &context_);
                } else {
                  math::Col2ImNd<T, Context, StorageOrder::NCHW>(
                      kernel_.size(),
                      input_offset,
                      col_buffer_size,
                      img_shape.data(),
                      col_buffer_shape.data(),
                      kernel_.data(),
                      stride_.data(),
                      dilation_.data(),
                      pads_.data(),
                      col_buffer_data,
                      dXdata,
                      &context_);
                }
                dXdata += input_offset;
                dYdata += output_offset;
              }
            }
          }
          return true;
        */
    }

    #[inline] pub fn run_on_device_with_orderNHWC(&mut self) -> bool {
        todo!();
        /*
            auto& X = Input(INPUT);
          auto& filter = Input(FILTER);
          auto& dY = Input(OUTPUT_GRAD);

          const int N = X.dim32(0), C = X.dim32(X.dim() - 1);

          const vector<int> input_dims = this->GetDims(X);
          const int input_image_size = this->GetDimsSize(X);

          const vector<int> output_dims = this->GetDims(dY);
          // The output image size is the spatial size of the output.
          const int output_image_size = this->GetDimsSize(dY);

          ConvPoolOpBase<Context>::ComputePads(input_dims);
          CAFFE_ENFORCE_EQ(X.dim(), filter.dim());
          const int M = filter.dim32(0);
          CAFFE_ENFORCE_EQ(C, filter.dim32(filter.dim() - 1) * group_);

          int kernel_dims_size = 1;
          for (size_t i = 0; i < kernel_.size(); ++i) {
            CAFFE_ENFORCE_EQ(filter.dim32(i + 1), kernel_[i]);
            kernel_dims_size *= kernel_[i];
          }

          CAFFE_ENFORCE_EQ(M % group_, 0);
          auto* dfilter = Output(FILTER_GRAD, filter.sizes(), at::dtype<T>());
          // The dimension of each kernel
          const int kernel_dim = C / group_ * kernel_dims_size;

          // The col buffer is stored in HWC order as well - the height and width, and
          // kernel_dim.
          vector<int> img_shape(X.sizes().cbegin() + 1, X.sizes().cend());
          vector<int> col_buffer_shape(output_dims.size() + 1);
          std::copy(output_dims.cbegin(), output_dims.cend(), col_buffer_shape.begin());
          col_buffer_shape.back() = C * kernel_dims_size;
          vector<int64_t> col_buffer_shape_64;
          std::copy(
              col_buffer_shape.cbegin(),
              col_buffer_shape.cend(),
              std::back_inserter(col_buffer_shape_64));
          ReinitializeTensor(
              &col_buffer_,
              col_buffer_shape_64,
              at::dtype<T>().device(Context::GetDeviceType()));

          if (kernel_.size() != 2) {
            SetDeviceTensor(img_shape, &img_shape_device_);
            SetDeviceTensor(col_buffer_shape, &col_buffer_shape_device_);
          }

          const int col_buffer_size = C * kernel_dims_size * output_image_size;
          const T* Xdata = X.template data<T>();
          const T* const filter_data = filter.template data<T>();
          const T* const dYdata = dY.template data<T>();
          T* col_buffer_data = col_buffer_.template mutable_data<T>();
          T* dfilter_data = dfilter->template mutable_data<T>();

          // Pre-setting the gradients to zero.
          math::Set<T, Context>(dfilter->numel(), 0, dfilter_data, &context_);

          T* dbias_data = nullptr;
          if (!no_bias_) {
            auto* dbias = Output(BIAS_OR_INPUT_GRAD, {M}, at::dtype<T>());
            dbias_data = dbias->template mutable_data<T>();
            math::Set<T, Context>(dbias->numel(), 0, dbias_data, &context_);
            // Removed the check for whether bias_multiplier_ has correct size or not
            ReinitializeTensor(
                &bias_multiplier_,
                vector<int64_t>(1, output_image_size),
                at::dtype<T>().device(Context::GetDeviceType()));
            math::Set<T, Context>(
                output_image_size,
                static_cast<T>(1),
                bias_multiplier_.template mutable_data<T>(),
                &context_);
          }

          if (N == 0) {
            if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
              auto* dX = Output(
                  no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD,
                  X.sizes(),
                  at::dtype<T>());
              dX->template mutable_data<T>();
            }
            return true;
          }

          // The offset corresponding to a single input image, and a single output
          // image.
          const size_t input_offset = C * input_image_size;
          const size_t output_offset = dY.numel() / dY.dim32(0);
          for (int image_id = 0; image_id < N; ++image_id) {
            // When we compute the gradient with respect to the filters, we need to do
            // im2col to allow gemm-type computation.
            if (kernel_.size() <= 2) {
              math::Im2Col<T, Context, StorageOrder::NHWC>(
                  C,
                  X.size(1),
                  kernel_.size() == 2 ? X.dim32(2) : 1,
                  kernel_h(),
                  kernel_.size() == 2 ? kernel_w() : 1,
                  dilation_h(),
                  kernel_.size() == 2 ? dilation_w() : 1,
                  pad_t(),
                  kernel_.size() == 2 ? pad_l() : 0,
                  kernel_.size() == 2 ? pad_b() : pad_l(),
                  kernel_.size() == 2 ? pad_r() : 0,
                  stride_h(),
                  kernel_.size() == 2 ? stride_w() : 1,
                  Xdata,
                  col_buffer_data,
                  &context_,
                  group_);
            } else {
              math::Im2ColNd<T, Context, StorageOrder::NHWC>(
                  kernel_.size(),
                  C * input_image_size,
                  col_buffer_size,
                  img_shape.data(),
                  col_buffer_shape.data(),
                  kernel_.data(),
                  stride_.data(),
                  dilation_.data(),
                  pads_.data(),
                  Xdata,
                  col_buffer_data,
                  &context_,
                  group_);
            }
            // Gradient with respect to filter.
            for (int group_id = 0; group_id < group_; ++group_id) {
              math::GemmEx<T, Context>(
                  CblasTrans,
                  CblasNoTrans,
                  M / group_,
                  kernel_dim,
                  output_image_size,
                  1,
                  dYdata + output_offset * image_id + group_id * (M / group_),
                  M,
                  col_buffer_data + group_id * kernel_dim,
                  group_ * kernel_dim,
                  1,
                  dfilter_data + group_id * (M / group_) * kernel_dim,
                  kernel_dim,
                  &context_);
            }
            if (!no_bias_) {
              // Gradient with respect to bias
              math::Gemv<T, Context>(
                  CblasTrans,
                  output_image_size,
                  M,
                  1,
                  dYdata + output_offset * image_id,
                  bias_multiplier_.template data<T>(),
                  1,
                  dbias_data,
                  &context_);
            }
            Xdata += input_offset;
          } // for each image

          if (OutputSize() == 3 || (no_bias_ && (OutputSize() == 2))) {
            // Compute the gradient w.r.t. the input.

            auto* dX = Output(
                no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD, X.sizes(), at::dtype<T>());
            T* dXdata = dX->template mutable_data<T>();
            for (int image_id = 0; image_id < N; ++image_id) {
              // Compute gradient into col_buffer.
              for (int group_id = 0; group_id < group_; ++group_id) {
                math::GemmEx<T, Context>(
                    CblasNoTrans,
                    CblasNoTrans,
                    output_image_size,
                    kernel_dim,
                    M / group_,
                    1,
                    dYdata + output_offset * image_id + group_id * (M / group_),
                    M,
                    filter_data + group_id * (M / group_) * kernel_dim,
                    kernel_dim,
                    0,
                    col_buffer_data + group_id * kernel_dim,
                    group_ * kernel_dim,
                    &context_);
              }
              if (kernel_.size() <= 2) {
                math::Col2Im<T, Context, StorageOrder::NHWC>(
                    C,
                    X.size(1),
                    kernel_.size() == 2 ? X.dim32(2) : 1,
                    kernel_h(),
                    kernel_.size() == 2 ? kernel_w() : 1,
                    dilation_h(),
                    kernel_.size() == 2 ? dilation_w() : 1,
                    pad_t(),
                    kernel_.size() == 2 ? pad_l() : 0,
                    kernel_.size() == 2 ? pad_b() : pad_l(),
                    kernel_.size() == 2 ? pad_r() : 0,
                    stride_h(),
                    kernel_.size() == 2 ? stride_w() : 1,
                    col_buffer_data,
                    dXdata,
                    &context_,
                    group_);
              } else {
                math::Col2ImNd<T, Context, StorageOrder::NHWC>(
                    kernel_.size(),
                    C * input_image_size,
                    col_buffer_size,
                    img_shape.data(),
                    col_buffer_shape.data(),
                    kernel_.data(),
                    stride_.data(),
                    dilation_.data(),
                    pads_.data(),
                    col_buffer_data,
                    dXdata,
                    &context_,
                    group_);
              }
              dXdata += input_offset;
            } // for each image
          }
          return true;
        */
    }
}
