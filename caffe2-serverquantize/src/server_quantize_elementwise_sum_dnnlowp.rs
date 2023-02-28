crate::ix!();

///---------------------------
register_cpu_operator_with_engine!{
    Sum, 
    DNNLOWP, 
    SumDNNLowPOp<u8, false>
}

///---------------------------
register_cpu_operator_with_engine!{
    SumRelu, 
    DNNLOWP, 
    SumDNNLowPOp<u8, true> 
}

num_inputs!{SumRelu, (1,INT_MAX)}

num_outputs!{SumRelu, 1}

inputs!{SumRelu, 
    0 => ("data_0", "First of the input tensors. Can be inplace.")
}

outputs!{SumRelu, 
    0 => ("sum", "Output tensor. Same dimension as inputs.")
}

identical_type_and_shape_of_input!{SumRelu, 0}

inputs_can_cross_devices!{SumRelu}

allow_inplace!{SumRelu, vec![(0, 0)]}

///---------------------------
register_cpu_operator_with_engine!{
    Int8Sum, 
    DNNLOWP, 
    SumDNNLowPOp<u8, false>
}

///---------------------------
register_cpu_operator_with_engine!{
    Int8SumRelu, 
    DNNLOWP, 
    SumDNNLowPOp<u8, true>
}

///---------------------------
register_cpu_operator_with_engine!{
    Sum, 
    DNNLOWP_16, 
    SumDNNLowPOp<u16, false>
}

///---------------------------
register_cpu_operator_with_engine!{
    SumRelu, 
    DNNLOWP_16, 
    SumDNNLowPOp<u16, true>
}

impl<T, const ReluFused: bool> SumDNNLowPOp<T, ReluFused> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : BaseType(operator_def, ws)
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

    #ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
      chrono::time_point<chrono::system_clock> t_begin, t_end;

      t_begin = chrono::system_clock::now();
    #endif

      if (!GetQuantizationParameters_()) {
        return false;
      }

    #ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
      t_end = chrono::system_clock::now();
      double dt = chrono::duration<double>(t_end - t_begin).count();
      LOG(INFO) << "this=" << this << " get_quant_params: " << dt * 1e3 << " ms";

      t_begin = chrono::system_clock::now();
    #endif

      using namespace dnnlowp;
      // Quantize inputs
      int len = InputTensorCPU_(0).size();

      // Element-wise sum
      int32_t intermediate_zero_point =
          intermediate_qparams_.zero_point * InputSize();

      auto* output = OutputTensorCPU_(0);
      output->ResizeLike(InputTensorCPU_(0));

      T* output_data = GetQuantizedOutputData_();

      if (InputTensorCPU_(0).template IsType<T>()) {
        if (InputSize() == 2 && is_same<T, uint8_t>::value && GetCpuId().avx2() &&
            GetCpuId().fma()) {
          // fast path when we have 2 uint8_t inputs with AVX2 / FMA support
          // NOTE: this path does addition in floating point unlike slow path that
          // does everything in fixed-point. So they are numerically different.
          array<const T*, 2> input_data;
          for (int i = 0; i < 2; ++i) {
            input_data[i] = InputTensorCPU_(i).template data<T>();
          }

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
          {
            constexpr int VLEN = 8;
            int j_begin, j_end;
            tie(j_begin, j_end) = Get1DPartition(
                len, dnnlowp_get_num_threads(), dnnlowp_get_thread_num(), VLEN);

            internal::ElementWiseSumAVX2<T, ReluFused>(
                input_data[0] + j_begin,
                input_data[1] + j_begin,
                output_data + j_begin,
                j_end - j_begin,
                in_qparams_[0].scale,
                in_qparams_[0].zero_point,
                in_qparams_[1].scale,
                in_qparams_[1].zero_point,
                out_qparams_.scale,
                out_qparams_.zero_point);
          } // omp parallel
        } else {
          vector<RequantizationParams> in_requantization_params(InputSize());
          vector<T*> input_data(InputSize());
          for (int i = 0; i < InputSize(); ++i) {
            float real_multiplier =
                in_qparams_[i].scale / intermediate_qparams_.scale;
            in_requantization_params[i] = qfactory_->ChooseRequantizationMultiplier(
                real_multiplier, intermediate_qparams_);
            input_data[i] = InputTensorCPU_(i).template data<T>();
          }

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
          {
            int j_begin, j_end;
            tie(j_begin, j_end) = Get1DPartition(
                len, dnnlowp_get_num_threads(), dnnlowp_get_thread_num());

            for (int j = j_begin; j < j_end; ++j) {
              int32_t acc = 0;
              for (int i = 0; i < InputSize(); ++i) {
                acc += fbgemm::Requantize<int32_t>(
                    input_data[i][j] - in_qparams_[i].zero_point,
                    in_requantization_params[i]);
              }
              int32_t raw = acc - intermediate_zero_point;
              if (ReluFused) {
                raw = std::max(0, raw);
              }
              output_data[j] =
                  fbgemm::Requantize<T>(raw, out_requantization_params_);
            }
          }
        }
      } else { // InputTensorCPU_(0).template IsType<T>()
        vector<float*> input_data(InputSize());
        for (int i = 0; i < InputSize(); ++i) {
          input_data[i] = InputTensorCPU_(i).template data<float>();
        }

    #ifdef _OPENMP
    #pragma omp parallel
    #endif
        {
          int j_begin, j_end;
          tie(j_begin, j_end) = Get1DPartition(
              len, dnnlowp_get_num_threads(), dnnlowp_get_thread_num());

          for (int j = j_begin; j < j_end; ++j) {
            int32_t acc = 0;
            for (int i = 0; i < InputSize(); ++i) {
              acc += fbgemm::Quantize<int32_t>(
                  input_data[i][j],
                  intermediate_qparams_.zero_point,
                  intermediate_qparams_.scale,
                  qfactory_->GetEltwiseQuantizePrecision());
            }
            int32_t raw = acc - intermediate_zero_point;
            if (ReluFused) {
              raw = std::max(0, raw);
            }
            output_data[j] = fbgemm::Requantize<T>(raw, out_requantization_params_);
          }
        }
      } // !InputTensorCPU_(0).template IsType<T>()

    #ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
      t_end = chrono::system_clock::now();
      dt = chrono::duration<double>(t_end - t_begin).count();
      LOG(INFO) << "this=" << this << " requantize inputs: " << dt * 1e3 << " ms";

      t_begin = chrono::system_clock::now();
    #endif

      RunOnDeviceEpilogue_();

    #ifdef DNNLOWP_MEASURE_TIME_BREAKDOWN
      t_end = chrono::system_clock::now();
      dt = chrono::duration<double>(t_end - t_begin).count();
      LOG(INFO) << "this=" << this << " prologue: " << dt * 1e3 << " ms";

      t_begin = chrono::system_clock::now();
    #endif

      return true;
        */
    }
    
    #[inline] pub fn get_quantization_parameters(&mut self) -> bool {
        
        todo!();
        /*
            using namespace dnnlowp;

      // Find global min and max of all inputs
      float global_min = float::max,
            global_max = numeric_limits<float>::lowest();

      for (int i = 0; i < InputSize(); ++i) {
        in_qparams_[i] =
            GetInputTensorQuantizationParamsOf(this, i, qfactory_.get());

        global_min = std::min(global_min, in_qparams_[i].Min());
        global_max = std::max(global_max, in_qparams_[i].Max());
      }

      intermediate_qparams_ = qfactory_->ChooseQuantizationParams(
          global_min,
          global_max,
          qfactory_->GetEltwiseQuantizePrecision(),
          qfactory_->GetPreserveActivationSparsity());

      GetOutputQuantizationParams_();

      // requantize from the intermediate precision to the final precision
      float real_multiplier = intermediate_qparams_.scale / out_qparams_.scale;
      out_requantization_params_ =
          qfactory_->ChooseRequantizationMultiplier(real_multiplier, out_qparams_);

      return true;
        */
    }
}
