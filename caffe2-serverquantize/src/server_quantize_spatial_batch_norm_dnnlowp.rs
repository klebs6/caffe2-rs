crate::ix!();

/**
  | Note this implementation assumes SCALE,
  | BIAS,
  | 
  | EST_MEAN, and EST_VAR inputs are still
  | in fp32, so is epsilon argument
  |
  */
pub struct SpatialBNDNNLowPOp<T,const ReluFused: bool> {
    //USE_OPERATOR_FUNCTIONS(CPUContext);
    //USE_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, SpatialBNOp<CPUContext>);
    base: DNNLowPOp<T, SpatialBNOp<CPUContext>>,

    epsilon:  f64,
    order:    StorageOrder,
    alpha:    Tensor,
    beta:     Tensor,
}

input_tags!{
    SpatialBNDNNLowPOp
    {
        Input,
        Scale,
        Bias,
        EstMean,
        EstVar
    }
}

output_tags!{
    SpatialBNDNNLowPOp
    {
        Output
    }
}

#[inline] pub fn spatial_bn_nhwc_avx2<T>(
    n:              i32,
    c:              i32,
    hxw:            i32,
    in_zero_point:  i32,
    out_zero_point: i32,
    x:              *const T,
    alpha:          *const f32,
    beta:           *const f32,
    y:              *mut T,
    relu_fused:     bool)  {

    todo!();
    /*
    
    */
}

impl<T,const ReluFused: bool> SpatialBNDNNLowPOp<T,ReluFused> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : DNNLowPOp<T, SpatialBNOp<CPUContext>>(operator_def, ws),
          OP_SINGLE_ARG(double, "epsilon", epsilon_, 1e-5),
          order_(StringToStorageOrder( this->template GetSingleArgument<std::string>("order", "NCHW"))) 

      bool is_test = this->template GetSingleArgument<bool>("is_test", false);
      OPERATOR_NEEDS_FEATURE(
          is_test, "SpatialBN DNNLOWP op only works for inference.");
      CAFFE_ENFORCE_NE(
          order_,
          StorageOrder::UNKNOWN,
          "order should be either \"NCHW\" or \"NHWC\".");
      CAFFE_ENFORCE(OutputSize() == 1);
      CAFFE_ENFORCE_GT(epsilon_, 0);
        */
    }

    #[inline] pub fn compute_fused_param(
        &mut self, 
        c:     i32,
        scale: *const f32,
        bias:  *const f32,
        mean:  *const f32,
        var:   *const f32,
        alpha: *mut f32,
        beta:  *mut f32)  {
    
        todo!();
        /*
            EigenVectorArrayMap<float> alpha_arr(alpha, C);
      EigenVectorArrayMap<float> beta_arr(beta, C);
      alpha_arr = ConstEigenVectorArrayMap<float>(scale, C) *
          (ConstEigenVectorArrayMap<float>(var, C) + epsilon_).rsqrt();
      beta_arr = ConstEigenVectorArrayMap<float>(bias, C) -
          alpha_arr * ConstEigenVectorArrayMap<float>(mean, C);

      // Adjust alpha and beta considering quantization scales
      alpha_arr = alpha_arr * (in_qparams_[0].scale / out_qparams_.scale);
      beta_arr = beta_arr / out_qparams_.scale;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
    
        todo!();
        /*
            if (!this->arguments_parsed_) {
        dnnlowp::ParseDNNLowPOperatorArguments(
            this, &dequantize_output_, &measure_quantization_error_, &followed_by_);

        if (ReluFused) {
          // It's actually fused with Relu not followed by but setting this to make
          // sure quantization error is correctly measured in
          // this->MeasureQuantizationError_
          followed_by_ = "Relu";
          dnnlowp::AdjustOutputTensorQuantizationParamsWithFollowedBy(
              this, followed_by_);
        }
        this->arguments_parsed_ = true;
      }

      const auto& X = InputTensorCPU_(INPUT);
      const auto& scale = Input(SCALE);
      const auto& bias = Input(BIAS);

      const int ndim = X.dim();
      CAFFE_ENFORCE_GE(ndim, 3);
      const int N = X.dim32(0);
      const int C = (order_ == StorageOrder::NCHW ? X.dim32(1) : X.dim32(ndim - 1));
      const std::vector<int> X_dims(X.sizes().cbegin(), X.sizes().cend());
      const int HxW = X.size_from_dim(1) / C;
      CAFFE_ENFORCE_EQ(scale.numel(), C);
      CAFFE_ENFORCE_EQ(bias.numel(), C);

      GetOutputQuantizationParams_();

      in_qparams_[0] = GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());

      const float* scale_data = scale.template data<float>();
      const float* bias_data = bias.template data<float>();
      ReinitializeTensor(
          &alpha_, {C}, at::dtype<float>().device(CPUContext::GetDeviceType()));
      ReinitializeTensor(
          &beta_, {C}, at::dtype<float>().device(CPUContext::GetDeviceType()));
      float* alpha_data = alpha_.template mutable_data<float>();
      float* beta_data = beta_.template mutable_data<float>();
      const auto& mean = Input(EST_MEAN);
      const auto& var = Input(EST_VAR);
      CAFFE_ENFORCE_EQ(mean.numel(), C);
      CAFFE_ENFORCE_EQ(var.numel(), C);

      auto* Y = OutputTensorCPU_(OUTPUT);
      Y->Resize(X.sizes());
      T* Y_data = GetQuantizedOutputData_();
      if (N == 0) {
        return true;
      }

      ComputeFusedParam_(
          C,
          scale_data,
          bias_data,
          mean.template data<float>(),
          var.template data<float>(),
          alpha_data,
          beta_data);

      vector<T> X_temp;
      const T* X_data =
          dnnlowp::QuantizeInputIfNeeded(this, 0, in_qparams_[0], X_temp);

      if (order_ == StorageOrder::NCHW) {
        for (int c = 0; c < C; ++c) {
          for (int i = 0; i < N; ++i) {
            for (int j = 0; j < HxW; ++j) {
              long quantized_down = out_qparams_.zero_point +
                  std::lrintf(alpha_data[c] *
                                  (X_data[(i * C + c) * HxW + j] -
                                   in_qparams_[0].zero_point) +
                              beta_data[c]);
              if (ReluFused) {
                quantized_down =
                    std::max<long>(quantized_down, out_qparams_.zero_point);
              }
              Y_data[(i * C + c) * HxW + j] =
                  fbgemm::clamp<long, T>(quantized_down, 8);
            }
          }
        }
      } else {
        if (GetCpuId().avx2()) {
          internal::SpatialBNNHWCAVX2<T>(
              N,
              C,
              HxW,
              in_qparams_[0].zero_point,
              out_qparams_.zero_point,
              X_data,
              alpha_data,
              beta_data,
              Y_data,
              ReluFused);
        } else {
          for (int i = 0; i < N * HxW; ++i) {
            for (int c = 0; c < C; ++c) {
              long quantized_down = out_qparams_.zero_point +
                  std::lrintf(alpha_data[c] *
                                  (X_data[i * C + c] - in_qparams_[0].zero_point) +
                              beta_data[c]);
              if (ReluFused) {
                quantized_down =
                    std::max<long>(quantized_down, out_qparams_.zero_point);
              }
              Y_data[i * C + c] = fbgemm::clamp<long, T>(quantized_down, 8);
            }
          }
        }
      }

      RunOnDeviceEpilogue_();

      return true;
        */
    }
}

register_cpu_operator_with_engine!{
    SpatialBN,
    DNNLOWP,
    SpatialBNDNNLowPOp<u8>
}

register_cpu_operator_with_engine!{
    Int8SpatialBN,
    DNNLOWP,
    SpatialBNDNNLowPOp<u8>
}

register_cpu_operator_with_engine!{
    Int8SpatialBNRelu,
    DNNLOWP,
    SpatialBNDNNLowPOp<u8, true>
}

///-------
num_inputs!{Int8SpatialBN, 5}

num_outputs!{Int8SpatialBN, 1}

///-------
num_inputs!{Int8SpatialBNRelu, 5}

num_outputs!{Int8SpatialBNRelu, 1}

