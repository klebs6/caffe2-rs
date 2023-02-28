crate::ix!();

use crate::{
    GradientMakerBase,
    ConvPoolOpBase,
    Workspace,
    OperatorDef,
    Tensor
};

///--------------------------------------------------
pub struct DeformConvOpBase<T, Context> {
    //USE_CONV_POOL_BASE_FUNCTIONS(Context);
    base: ConvPoolOpBase<Context>,

    deformable_group: i32,

    phantom: PhantomData<T>,
}

#[macro_export] macro_rules! use_deformable_conv_base_functions {
  ($T:ident, $Context:ident) => {
      todo!();
      /*
      USE_CONV_POOL_BASE_FUNCTIONS(Context);                 
      using DeformConvOpBase<T, Context>::deformable_group_; 
      using DeformConvOpBase<T, Context>::DeformableIm2col;  
      using DeformConvOpBase<T, Context>::DeformableCol2im;  
      using DeformConvOpBase<T, Context>::DeformableCol2imCoord
      */
  }
}

impl<T,Context> DeformConvOpBase<T,Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : ConvPoolOpBase<Context>(operator_def, ws),
            deformable_group_(
                this->template GetSingleArgument<int>("deformable_group", 1))
        */
    }
    
    #[inline] pub fn deformable_im2col(
        &mut self, 
        data_im:       *const T,
        data_offset:   *const T,
        im_shape:      &[i32],
        col_shape:     &[i32],
        data_col:      *mut T)  
    {
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn deformable_col2im(
        &mut self, 
        data_col:      *const T,
        data_offset:   *const T,
        im_shape:      &[i32],
        col_shape:     &[i32],
        grad_im:       *mut T)  
    {
        todo!();
        /*
        
        */
    }
    
    #[inline] pub fn deformable_col2im_coord(
        &mut self, 
        data_col:      *const T,
        data_im:       *const T,
        data_offset:   *const T,
        im_shape:      &[i32],
        col_shape:     &[i32],
        grad_offset:   *mut T)  
    {
        todo!();
        /*
        
        */
    }
}

/**
  | Deformable convolution operator consumes
  | an input vector, the kernel offsets
  | blob, the filter blob and the bias blob
  | and computes the output.
  | 
  | Other parameters, such as the stride
  | and kernel size, or the pads' sizes in
  | each direction are not necessary for
  | input because they are provided by the
  | 
  | ConvPoolOpBase operator.
  | 
  | Various dimension checks are done implicitly,
  | and the sizes are specified in the Input
  | docs for this operator.
  | 
  | As is expected, the filter is convolved
  | with a subset of the image using the deformed
  | kernel as specified by offsets blob
  | and the bias is added; this is done throughout
  | the image data and the output is computed.
  |
  */
pub struct DeformConvOp<T, Context> {

    //USE_DEFORMABLE_CONV_BASE_FUNCTIONS(T, Context);
    base: DeformConvOpBase<T, Context>,

    col_buffer_: Tensor, //{Context::GetDeviceType()};

    bias_multiplier_: Tensor,

    img_shape_device_: Tensor, //{Context::GetDeviceType()};

    col_buffer_shape_device_: Tensor, //{Context::GetDeviceType()};

    /**
      | Input: X, o, W, b
      | 
      | Output: Y
      |
      */
    phantom: PhantomData<T>,
}

num_inputs!{DeformConv, (3,4)}

num_outputs!{DeformConv, 1}

inputs!{DeformConv, 
    0 => ("X",       "Input data blob from previous layer; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the NCHW usage. On the other hand, the NHWC Op has a different set of dimension constraints."),
    1 => ("offset",  "Offsets blob that specifies the deformed shape of the kernel; consists of 2d offsets for each kernel element, one full set per each output element; therefore has size (N x 2*kH*kW x H' x W') where N is the batch size, kH and kW are the height and width of the kernel, H' and W' are the output blob dimensions."),
    2 => ("filter",  "The filter blob that will be used in the convolutions; has size (M x C x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel."),
    3 => ("bias",    "The 1D bias blob that is added through the convolution; has size (M).")
}

outputs!{DeformConv, 
    0 => ("Y", "Output data blob that contains the result of the convolution. The output dimensions are functions of the kernel size, stride size, and pad lengths.")
}

tensor_inference_function!{
    DeformConv, 
    ConvPoolOpBase::<CPUContext>::TensorInferenceForConv
}

input_tags!{
    DeformConvOp {
        Input,
        Offset,
        Filter,
        Bias
    }
}

impl<T, Context> DeformConvOp<T, Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : DeformConvOpBase<T, Context>(operator_def, ws) 
        // Create shared buffer mutex in the constructor
        // to avoid race-condition in DAGNet.
        if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
          createSharedBuffer<Context>(ws_);
        }
        */
    }

    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();
        /*
            const Tensor& X = Input(INPUT);
      const Tensor& offset = Input(OFFSET);
      auto& filter = Input(FILTER);
      Tensor* Y = Output(0);
      const int N = X.dim32(0), C = X.dim32(1);
      CAFFE_ENFORCE_EQ(X.dim(), filter.dim());
      const int M = filter.dim32(0);
      CAFFE_ENFORCE(
          C == filter.dim32(1) * group_,
          "Convolution op: input channels does not match: # of input channels ",
          C,
          " is not equal to kernel channels * group:",
          filter.dim32(1),
          "*",
          group_);
      CAFFE_ENFORCE(
          M % group_ == 0,
          "The number of output channels is not divisible by group.");
      CAFFE_ENFORCE(
          kernel_.size() == 2,
          "Deformable convolution only supports 2d kernel, has ",
          kernel_.size(),
          "d kernel.");
      CAFFE_ENFORCE(
          offset.dim() == 4,
          "Deformable convolution only supports 4d offset, has ",
          offset.dim(),
          "d offset.");
      CAFFE_ENFORCE_EQ(offset.dim32(0), N);
      CAFFE_ENFORCE(
          C % deformable_group_ == 0,
          "The number of input channels ",
          C,
          " is not divisible by deformable group ",
          deformable_group_);
      CAFFE_ENFORCE(
          M % deformable_group_ == 0,
          "The number of output channels ",
          M,
          " is not divisible by deformable group ",
          deformable_group_);
      CAFFE_ENFORCE(
          offset.dim32(1) == 2 * kernel_h() * kernel_w() * deformable_group_,
          "Deformable convolution: offset 1st dimension must equal "
          "2 * kernel_h * kernel_w * deformable_group: 2 * ",
          kernel_h(),
          " * ",
          kernel_w(),
          " * ",
          deformable_group_);

      CAFFE_ENFORCE_EQ(
          offset.dim32(2),
          (X.dim32(2) + pad_t() + pad_b() - (dilation_h() * (kernel_h() - 1) + 1)) /
                  stride_h() +
              1);
      CAFFE_ENFORCE_EQ(
          offset.dim32(3),
          (X.dim32(3) + pad_l() + pad_r() - (dilation_w() * (kernel_w() - 1) + 1)) /
                  stride_w() +
              1);

      int kernel_dims_size = 1;
      for (int i = 0; i < kernel_.size(); ++i) {
        CAFFE_ENFORCE(filter.dim32(i + 2) == kernel_[i]);
        kernel_dims_size *= kernel_[i];
      }

      ConvPoolOpBase<Context>::SetOutputSize(X, Y, filter.dim32(0));

      const vector<int> input_dims = GetDims(X);
      const vector<int> output_dims = GetDims(*Y);
      const int input_image_size = this->GetDimsSize(X);
      const int output_image_size = this->GetDimsSize(*Y);

      vector<int> img_shape;
      img_shape.assign(X.sizes().begin() + 1, X.sizes().end());

      vector<int> buffer_shape;
      buffer_shape.push_back(C / group_ * kernel_dims_size);
      buffer_shape.insert(
          buffer_shape.end(), output_dims.begin(), output_dims.end());

      // The dimension of each kernel
      const int kernel_dim = C / group_ * kernel_dims_size;
      // The offset corresponding to a single input image, and a single output
      // image.
      const int input_offset = C / group_ * input_image_size;
      const int output_offset = M / group_ * output_image_size;
      const int offset_offset = offset.numel() / offset.dim32(0);
      const int filter_offset = filter.numel() / group_;

      // The col buffer is stored in CHW order as well - kernel_dim, and the height
      // and width.
      const T* Xdata = X.template data<T>();
      const T* offset_data = offset.template data<T>();

      if (InputSize() == 4) {
        auto& bias = Input(BIAS);
        CAFFE_ENFORCE(bias.dim() == 1);
        CAFFE_ENFORCE(bias.dim32(0) == M);
        if (bias_multiplier_.numel() != output_image_size) {
          // If the helper bias multiplier is not image size, reshape and fill it
          // with
          // one.
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
      }
      T* Ydata = Y->template mutable_data<T>();
      const T* bias_data = nullptr;
      if (InputSize() == 4) {
        bias_data = Input(BIAS).template data<T>();
      }

      auto f = [this,
                &filter_offset,
                &bias_data,
                &X,
                &buffer_shape,
                &N,
                &Xdata,
                &offset_data,
                &M,
                &filter,
                &output_image_size,
                &kernel_dim,
                &Ydata,
                &input_offset,
                &offset_offset,
                &output_offset](Tensor* col_buffer) {
        col_buffer->Resize(buffer_shape);
        T* col_buffer_data = col_buffer->template mutable_data<T>();
        // Im2col, followed by gemm.
        for (int image_id = 0; image_id < N; ++image_id) {
          for (int group_id = 0; group_id < group_; ++group_id) {
            DeformableIm2col(
                Xdata + group_id * input_offset,
                offset_data,
                X.sizes(),
                col_buffer->sizes(),
                col_buffer_data);
            // Weight term
            math::Gemm<T, Context>(
                CblasNoTrans,
                CblasNoTrans,
                M / group_,
                output_image_size,
                kernel_dim,
                1,
                filter.template data<T>() + group_id * filter_offset,
                col_buffer_data,
                0,
                Ydata + group_id * output_offset,
                &context_);
          }
          if (bias_data) {
            math::Gemm<T, Context>(
                CblasNoTrans,
                CblasNoTrans,
                M,
                output_image_size,
                1,
                1,
                bias_data,
                bias_multiplier_.template data<T>(),
                1,
                Ydata,
                &context_);
          }
          Xdata += input_offset * group_;
          Ydata += output_offset * group_;
          offset_data += offset_offset;
        }
      };

      if (FLAGS_caffe2_force_shared_col_buffer || shared_buffer_) {
        runWithSharedBuffer<Context>(ws_, f);
      } else {
        f(&col_buffer_);
      }
      return true;
        */
    }
}

///--------------------------------------------------
pub struct DeformConvGradientOp<T,Context> {

    //USE_DEFORMABLE_CONV_BASE_FUNCTIONS(T, Context);
    base: DeformConvOpBase<T, Context>,

    col_buffer:      Tensor,
    bias_multiplier: Tensor,

    img_shape_device: Tensor, // {Context::GetDeviceType()};

    col_buffer_shape_device: Tensor, // {Context::GetDeviceType()};

    no_bias_: bool,

    /**
      | input: X, W, dY
      | 
      | output: dO, dW, db, and optionally dX
      |
      */
    phantom: PhantomData<T>,
}

num_inputs!{DeformConvGradient, (4,4)}

num_outputs!{DeformConvGradient, (2,4)}

input_tags!{
    DeformConvGradientOp {
        Input,
        Offset,
        Filter,
        OutputGrad
    }
}

output_tags!{
    DeformConvGradientOp {
        OffsetGrad,
        FilterGrad,
        BiasOrInputGrad,
        InputGrad
    }
}

impl<T,Context> DeformConvGradientOp<T,Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : DeformConvOpBase<T, Context>(operator_def, ws),
            no_bias_(this->template GetSingleArgument<int>("no_bias", 0)) 
        CAFFE_ENFORCE(
            !(no_bias_ && OutputSize() == 4),
            "If bias is not present, you should not have 4 grad output.");
        */
    }
}

pub struct GetDeformConvGradient;

impl GetGradientDefs for GetDeformConvGradient {

    #[inline] fn get_gradient_defs(&mut self) -> Vec<OperatorDef> {
        
        todo!();
        /*
            CAFFE_ENFORCE(def_.input_size() == 3 || def_.input_size() == 4);

        ArgumentHelper argsHelper(def_);

        auto compute_dX =
            !argsHelper.GetSingleArgument<bool>("no_gradient_to_input", 0);

        if (def_.input_size() == 4) {
          if (compute_dX) {
            return SingleGradientDef(
                "DeformConvGradient",
                "",
                vector<string>{I(0), I(1), I(2), GO(0)},
                vector<string>{GI(1), GI(2), GI(3), GI(0)});
          } else {
            return SingleGradientDef(
                "DeformConvGradient",
                "",
                vector<string>{I(0), I(1), I(2), GO(0)},
                vector<string>{GI(1), GI(2), GI(3)});
          }
        } else {
          if (compute_dX) {
            return SingleGradientDef(
                "DeformConvGradient",
                "",
                vector<string>{I(0), I(1), I(2), GO(0)},
                vector<string>{GI(1), GI(2), GI(0)},
                vector<Argument>{MakeArgument<int>("no_bias", 1)});
          } else {
            return SingleGradientDef(
                "DeformConvGradient",
                "",
                vector<string>{I(0), I(1), I(2), GO(0)},
                vector<string>{GI(1), GI(2)},
                vector<Argument>{MakeArgument<int>("no_bias", 1)});
          }
        }
        */
    }
}

register_gradient!{
    DeformConv, 
    GetDeformConvGradient
}

impl<T, Context> DeformConvGradientOp<T, Context> {

    #[inline] pub fn run_on_device_with_orderNCHW(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(INPUT);
      auto& offset = Input(OFFSET);
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
      CAFFE_ENFORCE(filter.dim32(1) * group_ == C);

      CAFFE_ENFORCE(
          kernel_.size() == 2,
          "Deformable convolution only supports 2d kernel, has ",
          kernel_.size(),
          "d kernel.");
      CAFFE_ENFORCE(
          offset.dim() == 4,
          "Deformable convolution only supports 4d offset, has ",
          offset.dim(),
          "d offset.");
      CAFFE_ENFORCE_EQ(offset.dim32(0), N);
      CAFFE_ENFORCE(
          C % deformable_group_ == 0,
          "The number of input channels ",
          C,
          " is not divisible by deformable group ",
          deformable_group_);
      CAFFE_ENFORCE(
          M % deformable_group_ == 0,
          "The number of output channels ",
          M,
          " is not divisible by deformable group ",
          deformable_group_);
      CAFFE_ENFORCE(
          offset.dim32(1) == 2 * kernel_h() * kernel_w() * deformable_group_,
          "Deformable convolution: offset 1st dimension must equal "
          "2 * kernel_h * kernel_w * deformable_group: 2 * ",
          kernel_h(),
          " * ",
          kernel_w(),
          " * ",
          deformable_group_);

      CAFFE_ENFORCE_EQ(
          offset.dim32(2),
          (X.dim32(2) + pad_t() + pad_b() - (dilation_h() * (kernel_h() - 1) + 1)) /
                  stride_h() +
              1);
      CAFFE_ENFORCE_EQ(
          offset.dim32(3),
          (X.dim32(3) + pad_l() + pad_r() - (dilation_w() * (kernel_w() - 1) + 1)) /
                  stride_w() +
              1);

      int kernel_dims_size = 1;
      for (int i = 0; i < kernel_.size(); ++i) {
        CAFFE_ENFORCE(filter.dim32(i + 2) == kernel_[i]);
        kernel_dims_size *= kernel_[i];
      }

      CAFFE_ENFORCE(M % group_ == 0);
      auto* dfilter = Output(FILTER_GRAD, filter.sizes(), at::dtype<T>());
      auto* doffset = Output(OFFSET_GRAD, offset.sizes(), at::dtype<T>());

      // The dimension of each kernel
      const int kernel_dim = C / group_ * kernel_dims_size;
      // The offset corresponding to a single input image, and a single output
      // image.
      const int input_offset = C / group_ * input_image_size;
      const int output_offset = M / group_ * output_image_size;
      const int offset_offset = offset.numel() / offset.dim32(0);
      const int filter_offset = filter.numel() / group_;

      // The col buffer is stored in CHW order as well - kernel_dim, and the
      // height and width.
      vector<int64_t> img_shape;
      img_shape.assign(X.sizes().begin() + 1, X.sizes().end());
      vector<int64_t> col_buffer_shape;
      col_buffer_shape.push_back(C * kernel_dims_size);
      col_buffer_shape.insert(
          col_buffer_shape.end(), output_dims.begin(), output_dims.end());
      ReinitializeTensor(
          &col_buffer_,
          col_buffer_shape,
          at::dtype<T>().device(Context::GetDeviceType()));

      const int col_buffer_offset = col_buffer_.numel() / group_;

      const T* Xdata = X.template data<T>();
      const T* filter_data = filter.template data<T>();
      const T* offset_data = offset.template data<T>();
      const T* dYdata = dY.template data<T>();
      T* col_buffer_data = col_buffer_.template mutable_data<T>();
      T* dfilter_data = dfilter->template mutable_data<T>();
      T* doffset_data = doffset->template mutable_data<T>();

      // Pre-setting the gradients to zero.
      math::Set<T, Context>(dfilter->numel(), 0, dfilter_data, &context_);

      T* dbias_data = nullptr;
      if (!no_bias_) {
        auto* dbias = Output(BIAS_OR_INPUT_GRAD, {M}, at::dtype<T>());
        if (bias_multiplier_.numel() != output_image_size) {
          // If the helper bias multiplier is not M, reshape and fill it with one.
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
        dbias_data = dbias->template mutable_data<T>();
        math::Set<T, Context>(dbias->numel(), 0, dbias_data, &context_);
      }

      T* dXdata = nullptr;
      if (OutputSize() == 4 || (no_bias_ && (OutputSize() == 3))) {
        auto* dX = Output(
            no_bias_ ? BIAS_OR_INPUT_GRAD : INPUT_GRAD, X.sizes(), at::dtype<T>());
        dXdata = dX->template mutable_data<T>();
        math::Set<T, Context>(dX->numel(), 0, dXdata, &context_);
      }

      for (int image_id = 0; image_id < N; ++image_id) {
        for (int group_id = 0; group_id < group_; ++group_id) {
          math::Gemm<T, Context>(
              CblasTrans,
              CblasNoTrans,
              kernel_dim,
              output_image_size,
              M / group_,
              1,
              filter_data + group_id * filter_offset,
              dYdata + group_id * output_offset,
              0,
              col_buffer_data + group_id * col_buffer_offset,
              &context_);
        }

        // Gradient with respect to offsets
        DeformableCol2imCoord(
            col_buffer_data,
            Xdata,
            offset_data,
            X.sizes(),
            col_buffer_shape,
            doffset_data);

        // Gradient with respect to input data
        if (dXdata) {
          DeformableCol2im(
              col_buffer_data, offset_data, X.sizes(), col_buffer_shape, dXdata);
          dXdata += input_offset * group_;
        }

        // Gradient with respect to filter
        DeformableIm2col(
            Xdata, offset_data, X.sizes(), col_buffer_shape, col_buffer_data);

        for (int group_id = 0; group_id < group_; ++group_id) {
          math::Gemm<T, Context>(
              CblasNoTrans,
              CblasTrans,
              M / group_,
              kernel_dim,
              output_image_size,
              1,
              dYdata + group_id * output_offset,
              col_buffer_data + group_id * col_buffer_offset,
              1,
              dfilter_data + group_id * filter_offset,
              &context_);
        }

        // Gradient with respect to bias
        if (dbias_data) {
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
        offset_data += offset_offset;
        doffset_data += offset_offset;
      }

      return true;
        */
    }
}
