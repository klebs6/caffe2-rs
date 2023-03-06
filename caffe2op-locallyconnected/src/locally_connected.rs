crate::ix!();

/**
  | The locally connected operator consumes
  | an input vector, a N-D filter blob and
  | a bias blob and computes the output.
  | 
  | -----------
  | @note
  | 
  | other parameters, such as the stride
  | and kernel size, or the pads' sizes in
  | each direction are not necessary for
  | input because they are provided by the
  | ConvPoolOpBase operator. Various
  | dimension checks are done implicitly,
  | and the sizes are specified in the Input
  | docs for this operator. As is expected,
  | the filter is locally connected with
  | a subset of the image and the bias is added;
  | this is done throughout the image data
  | and the output is computed. As a side
  | note on the implementation layout:
  | locally_connected_op_impl.h is the
  | templated implementation of the locally_connected_op.h
  | file, which is why they are separate
  | files.
  |
  */
#[USE_CONV_POOL_BASE_FUNCTIONS("Context")]
pub struct LocallyConnectedOp<T, Context> {
    base:                     ConvPoolOpBase<Context>,
    bias_multiplier:          Tensor, //{Context::GetDeviceType()};

    // Buffer.
    column_buffer:            Tensor, //{Context::GetDeviceType()};
    column_transposed_buffer: Tensor, //{Context::GetDeviceType()};
    y_transposed_buffer:      Tensor, //{Context::GetDeviceType()};

    phantom:                  PhantomData<T>,
}

inputs!{LocallyConnected,
    1 => ("filter", "The filter blob that will be used in the locally connected op; has size (YH * YW * M x C x kH x kW) if order == NCHW else (YH * YW * M  * KH * KW * C), where YH and YW are the height and width of the output image, C is the number of channels, and kH and kW are the height and width of the kernel." ),
    2 => ("bias",   "The 1D bias blob that is added through the locally connected op; has size (YH * YW * M).")
}

outputs!{LocallyConnected,
    0 => ("Y", "Output data blob that contains the result of the locally connected op. The output dimensions are functions of the kernel size, stride size, and pad lengths.")
}

/**
  | Input: X, W, b
  | 
  | Output: Y
  |
  */
input_tags!{
    LocallyConnectedOp {
        Input,
        Filter,
        Bias
    }
}

impl<T, Context> LocallyConnectedOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<Context>(std::forward<Args>(args)...) 

        // Since this is the default locally connected implementation, we will
        // use CAFFE_ENFORCE instead of OPERATOR_NEEDS_FEATURE.
        CAFFE_ENFORCE(
            group_ == 1 || order_ == StorageOrder::NCHW,
            "Group locally connected only supports NCHW order right now.");
        */
    }

    #[inline] pub fn run_on_device_with_order_nchw(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(INPUT);
      const auto& filter = Input(FILTER);
      auto* Y = Output(0);
      const int image_ndim = X.dim() - 2;
      CAFFE_ENFORCE_EQ(X.dim() + image_ndim, filter.dim());
      lc_op_util::ShapeParams shape;
      shape.N = X.dim32(0);
      shape.C = X.dim32(1);
      shape.M = filter.dim32(image_ndim);
      CAFFE_ENFORCE(
          shape.C == filter.dim32(image_ndim + 1) * group_,
          "Locally Connected op: input channels does not match: "
          "# of input channels ",
          shape.C,
          " is not equal to kernel channels * group:",
          filter.dim32(image_ndim + 1),
          "*",
          group_);
      CAFFE_ENFORCE_EQ(
          shape.M % group_,
          0,
          "The number of output channels is not divisible by group.");

      ConvPoolOpBase<Context>::SetOutputSize(X, Y, shape.M);
      shape.input_image_size = GetDimsSize(X);
      shape.output_image_size = GetDimsSize(*Y);
      const std::vector<int> output_image_dims = GetDims(*Y);
      for (int i = 0; i < image_ndim; ++i) {
        CAFFE_ENFORCE_EQ(output_image_dims[i], filter.dim32(i));
      }

      int kernel_dims_size = 1;
      for (std::size_t i = 0; i < kernel_.size(); ++i) {
        CAFFE_ENFORCE_EQ(filter.dim32(i + image_ndim + 2), kernel_[i]);
        kernel_dims_size *= kernel_[i];
      }

      shape.X_dims.assign(X.sizes().cbegin() + 1, X.sizes().cend());
      shape.kernel_size = shape.C / group_ * kernel_dims_size;
      lc_op_util::SetColumnBufferShape(
          shape.N,
          shape.kernel_size,
          shape.output_image_size,
          output_image_dims,
          order_,
          &shape.column_slice_dims,
          &shape.column_dims,
          &shape.column_transposed_dims,
          &shape.column_axes);
      lc_op_util::SetYBufferShape(
          shape.N,
          shape.M,
          shape.output_image_size,
          order_,
          &shape.Y_dims,
          &shape.Y_transposed_dims,
          &shape.Y_axes);

      const T* X_data = X.template data<T>();
      const T* filter_data = filter.template data<T>();
      const T* bias_data = nullptr;
      if (InputSize() == 3) {
        const auto& bias = Input(BIAS);
        CAFFE_ENFORCE_EQ(bias.dim(), image_ndim + 1);
        for (int i = 0; i < image_ndim; ++i) {
          CAFFE_ENFORCE_EQ(bias.dim32(i), output_image_dims[i]);
        }
        CAFFE_ENFORCE_EQ(bias.dim32(image_ndim), shape.M);
        bias_data = bias.template data<T>();
        ConvPoolOpBase<Context>::template SetBiasMultiplier<T>(
            shape.N, &bias_multiplier_);
      }
      T* Y_data = Y->template mutable_data<T>();

      RunOnDeviceWithOrderNCHWImpl(
          shape,
          X_data,
          filter_data,
          bias_data,
          Y_data,
          &column_buffer_,
          &column_transposed_buffer_,
          &Y_transposed_buffer_);

      return true;
        */
    }

    #[inline] pub fn run_on_device_with_order_nhwc(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(INPUT);
      const auto& filter = Input(FILTER);
      auto* Y = Output(0);
      CAFFE_ENFORCE_EQ(
          kernel_.size(),
          2,
          "Only 2d locally connected op is supported for NHWC storage type.");
      const int image_ndim = X.dim() - 2;
      CAFFE_ENFORCE_EQ(X.dim() + image_ndim, filter.dim());
      lc_op_util::ShapeParams shape;
      shape.N = X.dim32(0);
      shape.C = X.dim32(3);
      shape.X_dims = {X.dim32(1), X.dim32(2), X.dim32(3)};
      shape.M = filter.dim32(image_ndim);
      CAFFE_ENFORCE_EQ(filter.dim32(image_ndim + 1), kernel_h());
      CAFFE_ENFORCE_EQ(filter.dim32(image_ndim + 2), kernel_w());
      CAFFE_ENFORCE_EQ(filter.dim32(image_ndim + 3), shape.C);
      ConvPoolOpBase<Context>::SetOutputSize(X, Y, shape.M);

      shape.input_image_size = GetDimsSize(X);
      shape.output_image_size = GetDimsSize(*Y);
      const std::vector<int> output_image_dims = GetDims(*Y);
      for (int i = 0; i < image_ndim; ++i) {
        CAFFE_ENFORCE_EQ(output_image_dims[i], filter.dim32(i));
      }

      shape.kernel_size = kernel_h() * kernel_w() * shape.C;
      lc_op_util::SetColumnBufferShape(
          shape.N,
          shape.kernel_size,
          shape.output_image_size,
          output_image_dims,
          order_,
          &shape.column_slice_dims,
          &shape.column_dims,
          &shape.column_transposed_dims,
          &shape.column_axes);
      lc_op_util::SetYBufferShape(
          shape.N,
          shape.M,
          shape.output_image_size,
          order_,
          &shape.Y_dims,
          &shape.Y_transposed_dims,
          &shape.Y_axes);

      const T* X_data = X.template data<T>();
      const T* filter_data = filter.template data<T>();
      const T* bias_data = nullptr;
      if (InputSize() == 3) {
        const auto& bias = Input(BIAS);
        CAFFE_ENFORCE_EQ(bias.dim(), image_ndim + 1);
        for (int i = 0; i < image_ndim; ++i) {
          CAFFE_ENFORCE_EQ(bias.dim32(i), output_image_dims[i]);
        }
        CAFFE_ENFORCE_EQ(bias.dim32(image_ndim), shape.M);
        bias_data = bias.template data<T>();
        ConvPoolOpBase<Context>::template SetBiasMultiplier<T>(
            shape.N, &bias_multiplier_);
      }
      T* Y_data = Y->template mutable_data<T>();

      RunOnDeviceWithOrderNHWCImpl(
          shape,
          X_data,
          filter_data,
          bias_data,
          Y_data,
          &column_buffer_,
          &column_transposed_buffer_,
          &Y_transposed_buffer_);

      return true;
        */
    }
    
    #[inline] pub fn run_on_device_with_order_nchw_impl(
        &mut self, 
        shape:                    &ShapeParams,
        x_data:                   *const T,
        filter_data:              *const T,
        bias_data:                *const T,
        y_data:                   *mut T,
        column_buffer:            *mut Tensor,
        column_transposed_buffer: *mut Tensor,
        y_transposed_buffer:      *mut Tensor)  
    {
        todo!();
        /*
            const int input_stride = shape.C / group_ * shape.input_image_size;
      const int column_stride = shape.kernel_size * shape.output_image_size;
      column_buffer->Resize(shape.column_dims);
      column_transposed_buffer->Resize(shape.column_transposed_dims);
      Y_transposed_buffer->Resize(shape.Y_transposed_dims);
      T* column_buffer_data = column_buffer->template mutable_data<T>();
      T* Y_transposed_buffer_data = Y_transposed_buffer->template mutable_data<T>();

      for (int image_id = 0; image_id < shape.N; ++image_id) {
        for (int group_id = 0; group_id < group_; ++group_id) {
          if (kernel_.size() == 2) {
            math::Im2Col<T, Context, StorageOrder::NCHW>(
                shape.C / group_,
                shape.X_dims[1],
                shape.X_dims[2],
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
                X_data + group_id * input_stride,
                column_buffer_data + group_id * column_stride,
                &context_);
          } else {
            math::Im2ColNd<T, Context, StorageOrder::NCHW>(
                kernel_.size(),
                shape.C * shape.input_image_size,
                column_stride,
                shape.X_dims.data(),
                shape.column_slice_dims.data(),
                kernel_.data(),
                stride_.data(),
                dilation_.data(),
                pads_.data(),
                X_data + group_id * input_stride,
                column_buffer_data + group_id * column_stride,
                &context_);
          }
        }
        X_data += input_stride * group_;
        column_buffer_data += column_stride * group_;
      }
      math::Transpose(
          shape.column_dims.size(),
          shape.column_dims.data(),
          shape.column_axes.data(),
          column_buffer->template data<T>(),
          column_transposed_buffer->template mutable_data<T>(),
          &context_);
      math::GemmStridedBatched(
          CblasNoTrans,
          CblasNoTrans,
          shape.output_image_size * group_,
          shape.M / group_,
          shape.N,
          shape.kernel_size,
          1.0f,
          filter_data,
          shape.M / group_ * shape.kernel_size,
          column_transposed_buffer->template data<T>(),
          shape.kernel_size * shape.N,
          0.0f,
          Y_transposed_buffer_data,
          shape.M / group_ * shape.N,
          &context_);
      if (bias_data != nullptr) {
        math::Gemm<T, Context>(
            CblasNoTrans,
            CblasNoTrans,
            shape.output_image_size * shape.M,
            shape.N,
            1,
            1.0,
            bias_data,
            bias_multiplier_.template data<T>(),
            1.0,
            Y_transposed_buffer_data,
            &context_);
      }
      math::Transpose(
          shape.Y_transposed_dims.size(),
          shape.Y_transposed_dims.data(),
          shape.Y_axes.data(),
          Y_transposed_buffer_data,
          Y_data,
          &context_);
        */
    }
    
    #[inline] pub fn run_on_device_with_order_nhwc_impl(
        &mut self, 
        shape:                     &ShapeParams,
        x_data:                    *const T,
        filter_data:               *const T,
        bias_data:                 *const T,
        y_data:                    *mut T,
        column_buffer:             *mut Tensor,
        column_transposed_buffer:  *mut Tensor,
        y_transposed_buffer:       *mut Tensor)  
    {
        todo!();
        /*
            const int input_stride = shape.C * shape.input_image_size;
      const int column_stride = shape.kernel_size * shape.output_image_size;
      column_buffer->Resize(shape.column_dims);
      column_transposed_buffer->Resize(shape.column_transposed_dims);
      Y_transposed_buffer->Resize(shape.Y_transposed_dims);
      T* column_buffer_data = column_buffer->template mutable_data<T>();
      T* Y_transposed_buffer_data = Y_transposed_buffer->template mutable_data<T>();
      for (int image_id = 0; image_id < shape.N; ++image_id) {
        math::Im2Col<T, Context, StorageOrder::NHWC>(
            shape.C,
            shape.X_dims[0],
            shape.X_dims[1],
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
            X_data + image_id * input_stride,
            column_buffer_data + image_id * column_stride,
            &context_);
      }
      math::Transpose(
          shape.column_dims.size(),
          shape.column_dims.data(),
          shape.column_axes.data(),
          column_buffer->template data<T>(),
          column_transposed_buffer->template mutable_data<T>(),
          &context_);
      math::GemmStridedBatched(
          CblasNoTrans,
          CblasTrans,
          shape.output_image_size,
          shape.N,
          shape.M,
          shape.kernel_size,
          1.0f,
          column_transposed_buffer->template data<T>(),
          shape.N * shape.kernel_size,
          filter_data,
          shape.kernel_size * shape.M,
          0.0f,
          Y_transposed_buffer_data,
          shape.N * shape.M,
          &context_);
      math::Transpose(
          shape.Y_transposed_dims.size(),
          shape.Y_transposed_dims.data(),
          shape.Y_axes.data(),
          Y_transposed_buffer_data,
          Y_data,
          &context_);
      if (bias_data != nullptr) {
        math::Gemm<T, Context>(
            CblasNoTrans,
            CblasNoTrans,
            shape.N,
            shape.output_image_size * shape.M,
            1,
            1.0f,
            bias_multiplier_.template data<T>(),
            bias_data,
            1.0f,
            Y_data,
            &context_);
      }
        */
    }
}
