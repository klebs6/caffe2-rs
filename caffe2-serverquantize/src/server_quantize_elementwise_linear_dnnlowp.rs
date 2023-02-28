crate::ix!();

pub type ElementwiseLinearFp32Op = ElementwiseLinearOp<f32,CPUContext,DefaultEngine>;

pub struct ElementwiseLinearDNNLowPOp<T> {
    //USE_OPERATOR_FUNCTIONS(CPUContext);
    //USE_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, ElementwiseLinearFp32Op);
    base: DNNLowPOp<T, ElementwiseLinearFp32Op>,

    axis:                   i32,
    requantization_params:  RequantizationParams,
    a_quantized:            Vec<T>,
}

register_cpu_operator_with_engine!{
    ElementwiseLinear,
    DNNLOWP,
    ElementwiseLinearDNNLowPOp<u8>
}

register_cpu_operator_with_engine!{
    Int8ElementwiseLinear,
    DNNLOWP,
    ElementwiseLinearDNNLowPOp<u8>
}

impl<T> ElementwiseLinearDNNLowPOp<T> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : BaseType(operator_def, ws),
          axis_(this->template GetSingleArgument<int>("axis", 1))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (!GetQuantizationParameters_()) {
        return false;
      }

      const auto& X = InputTensorCPU_(0);
      const auto& a = InputTensorCPU_(1);
      const auto& b = InputTensorCPU_(2);
      auto* Y = OutputTensorCPU_(0);

      const auto canonical_axis = X.canonical_axis_index(axis_);
      const int N = X.size_to_dim(canonical_axis);
      const int D = X.size_from_dim(canonical_axis);

      CAFFE_ENFORCE_EQ(a.ndim(), 1, a.ndim());
      CAFFE_ENFORCE_EQ(a.size(0), D, a.ndim());
      CAFFE_ENFORCE_EQ(b.ndim(), 1, b.ndim());
      CAFFE_ENFORCE_EQ(b.size(0), D, b.ndim());

      Y->ResizeLike(X);

      // Quantize X
      vector<T> X_temp;
      const T* X_quantized =
          QuantizeInputIfNeeded<T>(this, 0, in_qparams_[0], X_temp);

      // Quantize b
      vector<int32_t> b_quantized(b.numel());
      const float* b_data = b.template data<float>();
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
      for (int i = 0; i < b.numel(); ++i) {
        b_quantized[i] = fbgemm::Quantize<int32_t>(
            b_data[i],
            0,
            in_qparams_[0].scale * in_qparams_[1].scale,
            32,
            true /* signed */);
      }

      T* Y_quantized = GetQuantizedOutputData_();
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
      for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
          int32_t raw = (X_quantized[n * D + d] - in_qparams_[0].zero_point) *
                  (a_quantized_[d] - in_qparams_[1].zero_point) +
              b_quantized[d];
          Y_quantized[n * D + d] =
              fbgemm::Requantize<T>(raw, requantization_params_);
        }
      }

      RunOnDeviceEpilogue_();

      return true;
        */
    }
    
    #[inline] pub fn get_quantization_parameters(&mut self) -> bool {
        
        todo!();
        /*
            using namespace dnnlowp;

      // Choose quantization for X
      in_qparams_[0] = GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());

      // Quantize a
      if (a_quantized_.empty()) {
        const auto& a = InputTensorCPU_(1);
        in_qparams_[1] = qfactory_->ChooseQuantizationParams(
            a.template data<float>(), a.numel(), true /*weight*/);

        a_quantized_.resize(a.numel());
        fbgemm::Quantize<T>(
            a.template data<float>(),
            a_quantized_.data(),
            a_quantized_.size(),
            in_qparams_[1]);
      }

      GetOutputQuantizationParams_();

      float real_multiplier =
          in_qparams_[0].scale * in_qparams_[1].scale / out_qparams_.scale;
      requantization_params_ =
          qfactory_->ChooseRequantizationMultiplier(real_multiplier, out_qparams_);

      return true;
        */
    }
}
