crate::ix!();


pub struct LSTMUnitDNNLowPOp<T: PrimInt> {

    base: LSTMUnitOp<CPUContext>,

    drop_states:                      bool,
    sigmoid:                          Sigmoid<T>,
    tanh:                             Tanh<T>,
    h_in_qparams:                     TensorQuantizationParams,
    c_in_qparams:                     TensorQuantizationParams,
    g_in_qparams:                     TensorQuantizationParams,
    h_out_qparams:                    TensorQuantizationParams,
    c_out_qparams:                    TensorQuantizationParams,
    fp32_op:                          Box<OpWrapper<LSTMUnitOp<CPUContext>,T>>,
    dequantize_output:                bool, // default = false
    measure_quantization_error:       bool, // default = false
    qfactory:                         Box<QuantizationFactory>,
    cell_quantization_error_stats:    QuantizationErrorStats,
    hidden_quantization_error_stats:  QuantizationErrorStats,
    arguments_parsed:                 bool, // default = false
}

register_cpu_operator_with_engine!{
    LSTMUnit,
    DNNLOWP,
    LSTMUnitDNNLowPOp<u8>
}

register_cpu_operator_with_engine!{
    Int8LSTMUnit,
    DNNLOWP,
    LSTMUnitDNNLowPOp<u8>
}

impl<T: PrimInt> Drop for LSTMUnitDNNLowPOp<T> {
    fn drop(&mut self) {
        todo!();
        /* 
      if (measure_quantization_error_) {
        ReportQuantizationError(this, cell_quantization_error_stats_);
        ReportQuantizationError(this, hidden_quantization_error_stats_);
      }
 */
    }
}

impl<T: PrimInt> LSTMUnitDNNLowPOp<T> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : LSTMUnitOp<CPUContext>(operator_def, ws),
          drop_states_( this->template GetSingleArgument<bool>("drop_states", false)),
          qfactory_(GetQuantizationFactoryOf(this))
        */
    }
    
    #[inline] pub fn fp_32op(&mut self) -> *mut OpWrapper<LSTMUnitOp<CPUContext>,T> {
    
        todo!();
        /*
            if (!fp32_op_) {
        fp32_op_.reset(
            new OpWrapper<LSTMUnitOp<CPUContext>, T>(this, qfactory_.get()));
      }
      return fp32_op_.get();
        */
    }
    
    #[inline] pub fn input_tensorcpu(&mut self, idx: i32) -> &TensorCPU {
        
        todo!();
        /*
            return InputIsType<int8::Int8TensorCPU>(idx)
          ? this->template Input<int8::Int8TensorCPU>(idx).t
          : Input(idx);
        */
    }
    
    #[inline] pub fn output_tensorcpu(&mut self, idx: i32) -> *mut TensorCPU {
        
        todo!();
        /*
            if (dequantize_output_) {
        return Output(idx);
      } else {
        return &Outputs()[idx]->template GetMutable<int8::Int8TensorCPU>()->t;
      }
        */
    }
    
    #[inline] pub fn get_quantization_parameters(&mut self) -> bool {
        
        todo!();
        /*
            using namespace dnnlowp;

      H_in_qparams_ =
          GetInputTensorQuantizationParamsOf(this, HIDDEN_T_M_1, qfactory_.get());
      C_in_qparams_ =
          GetInputTensorQuantizationParamsOf(this, CELL_T_M_1, qfactory_.get());

      // G is only used as an input to tanh or sigmoid
      G_in_qparams_ = qfactory_->ChooseQuantizationParams(
          std::min(
              sigmoid_.GetInputQuantizationParams().Min(),
              tanh_.GetInputQuantizationParams().Min()),
          std::max(
              sigmoid_.GetInputQuantizationParams().Max(),
              tanh_.GetInputQuantizationParams().Max()));

      if (HasStaticQuantization(this, HIDDEN_T)) {
        H_out_qparams_ = GetStaticQuantizationParamsOf(this, HIDDEN_T);
      }
      if (HasStaticQuantization(this, CELL_T)) {
        C_out_qparams_ = GetStaticQuantizationParamsOf(this, CELL_T);
      }

      if (!HasStaticQuantization(this, HIDDEN_T) ||
          !HasStaticQuantization(this, CELL_T)) {
        Fp32Op_()->DequantizeInput();
        if (!Fp32Op_()->Get()->RunOnDevice()) {
          return false;
        }
        if (!HasStaticQuantization(this, HIDDEN_T)) {
          H_out_qparams_ =
              Fp32Op_()->GetOutputQuantizationParams(qfactory_.get(), HIDDEN_T);
        }
        if (!HasStaticQuantization(this, CELL_T)) {
          C_out_qparams_ =
              Fp32Op_()->GetOutputQuantizationParams(qfactory_.get(), CELL_T);
        }
      }

      return true;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (!arguments_parsed_) {
        ParseDNNLowPOperatorArguments(
            this, &dequantize_output_, &measure_quantization_error_);
        arguments_parsed_ = true;
      }

      GetQuantizationParameters_();

      // Extract N
      const auto N = InputTensorCPU_(CELL_T_M_1).size(1);

      // Gates: 1xNxG
      const auto G = InputTensorCPU_(GATES).size(2);
      const auto D = InputTensorCPU_(CELL_T_M_1).size(2);

      CAFFE_ENFORCE_EQ(4 * D, G);

      // Quantize H_prev if needed
      vector<T> H_prev_temp;
      const T* H_prev =
          QuantizeInputIfNeeded(this, HIDDEN_T_M_1, H_in_qparams_, H_prev_temp);

      // Quantize C_prev if needed
      vector<T> C_prev_temp;
      const T* C_prev =
          QuantizeInputIfNeeded(this, CELL_T_M_1, C_in_qparams_, C_prev_temp);

      // Quantize X if needed
      vector<T> X_temp;
      const T* X = QuantizeInputIfNeeded(this, GATES, G_in_qparams_, X_temp);
      // first 3D input to sigmoid, last D input to tanh

      const size_t TIMESTEP = SEQ_LENGTHS + 1;

      CAFFE_ENFORCE_EQ(Input(SEQ_LENGTHS).size(), N);
      const auto* seqLengths = Input(SEQ_LENGTHS).template data<int32_t>();
      const auto t = static_cast<OperatorStorage*>(this)
                         ->Input<Tensor>(TIMESTEP, CPU)
                         .template data<int32_t>()[0];
      OutputTensorCPU_(CELL_T)->ResizeLike(InputTensorCPU_(CELL_T_M_1));
      OutputTensorCPU_(HIDDEN_T)->ResizeLike(InputTensorCPU_(CELL_T_M_1));

      vector<uint8_t> Ctemp, Htemp;
      uint8_t *Cdata, *Hdata;
      if (dequantize_output_) {
        Ctemp.resize(OutputTensorCPU_(CELL_T)->size());
        Cdata = Ctemp.data();

        Htemp.resize(OutputTensorCPU_(HIDDEN_T)->size());
        Hdata = Htemp.data();
      } else {
        Cdata = OutputTensorCPU_(CELL_T)->template mutable_data<uint8_t>();
        Hdata = OutputTensorCPU_(HIDDEN_T)->template mutable_data<uint8_t>();
      }

      int32_t forget_bias_quantized =
          fbgemm::Quantize<int32_t>(forget_bias_, G_in_qparams_);

      LSTMUnit(
          N,
          D,
          t,
          H_prev,
          C_prev,
          X,
          seqLengths,
          drop_states_,
          Cdata,
          Hdata,
          forget_bias_quantized,
          sigmoid_,
          tanh_,
          G_in_qparams_,
          C_in_qparams_,
          C_out_qparams_,
          H_in_qparams_,
          H_out_qparams_,
          qfactory_.get());

      if (dequantize_output_) {
        fbgemm::Dequantize<T>(
            Cdata,
            OutputTensorCPU_(CELL_T)->template mutable_data<float>(),
            Ctemp.size(),
            C_out_qparams_);
        fbgemm::Dequantize<T>(
            Hdata,
            OutputTensorCPU_(HIDDEN_T)->template mutable_data<float>(),
            Htemp.size(),
            H_out_qparams_);

        if (measure_quantization_error_) {
          MeasureQuantizationError(
              OutputTensorCPU_(CELL_T)->template mutable_data<float>(),
              Fp32Op_()->Get()->Output(CELL_T)->template data<float>(),
              OutputTensorCPU_(CELL_T)->size(),
              &cell_quantization_error_stats_);

          MeasureQuantizationError(
              OutputTensorCPU_(HIDDEN_T)->template mutable_data<float>(),
              Fp32Op_()->Get()->Output(HIDDEN_T)->template data<float>(),
              OutputTensorCPU_(HIDDEN_T)->size(),
              &hidden_quantization_error_stats_);
        }
      } else {
        PropagateOutputTensorQuantizationParams(this, HIDDEN_T, H_out_qparams_);
        PropagateOutputTensorQuantizationParams(this, CELL_T, C_out_qparams_);
      }

      return true;
        */
    }
}

#[inline] pub fn lstm_unit<T>(
    n:             i32,
    d:             i32,
    t:             i32,
    h_prev:        *const T,
    c_prev:        *const T,
    x:             *const T,
    seq_lengths:   *const i32,
    drop_states:   bool,
    c:             *mut T,
    h:             *mut T,
    forget_bias:   i32,
    sigmoid:       &Sigmoid<T>,
    tanh_:         &Tanh<T>,
    x_qparams:     &TensorQuantizationParams,
    c_in_qparams:  &TensorQuantizationParams,
    c_out_qparams: &TensorQuantizationParams,
    h_in_qparams:  &TensorQuantizationParams,
    h_out_qparams: &TensorQuantizationParams,
    qfactory:      *mut QuantizationFactory)  {

    todo!();
    /*
        const TensorQuantizationParams sigmoid_in_qparams =
          sigmoid.GetInputQuantizationParams();
      const TensorQuantizationParams sigmoid_out_qparams =
          sigmoid.GetOutputQuantizationParams();
      const TensorQuantizationParams tanh_in_qparams =
          tanh.GetInputQuantizationParams();
      const TensorQuantizationParams tanh_out_qparams =
          tanh.GetOutputQuantizationParams();

      RequantizationParams h_in_to_out_params =
          qfactory->ChooseRequantizationMultiplier(
              H_in_qparams.scale / H_out_qparams.scale, H_out_qparams);

      RequantizationParams c_in_to_out_params =
          qfactory->ChooseRequantizationMultiplier(
              C_in_qparams.scale / C_out_qparams.scale, C_out_qparams);

      float sigmoid_scale = sigmoid_out_qparams.scale;
      float tanh_scale = tanh_out_qparams.scale;
      int32_t sigmoid_zero_point = sigmoid_out_qparams.zero_point;
      int32_t tanh_zero_point = tanh_out_qparams.zero_point;

      RequantizationParams x_to_sigmoid_params =
          qfactory->ChooseRequantizationMultiplier(
              X_qparams.scale / sigmoid_in_qparams.scale, sigmoid_in_qparams);

      RequantizationParams x_to_tanh_params =
          qfactory->ChooseRequantizationMultiplier(
              X_qparams.scale / tanh_in_qparams.scale, tanh_in_qparams);

      RequantizationParams c_to_tanh_params =
          qfactory->ChooseRequantizationMultiplier(
              C_in_qparams.scale / tanh_scale, tanh_out_qparams);

      RequantizationParams c_out_requantization_params =
          qfactory->ChooseRequantizationMultiplier(
              sigmoid_scale * tanh_scale / C_out_qparams.scale, C_out_qparams);

      RequantizationParams c_tanh_requantization_params =
          qfactory->ChooseRequantizationMultiplier(
              sigmoid_scale * tanh_scale / tanh_in_qparams.scale, tanh_in_qparams);

      RequantizationParams h_requantization_params =
          qfactory->ChooseRequantizationMultiplier(
              sigmoid_scale * tanh_scale / H_out_qparams.scale, H_out_qparams);

      for (int n = 0; n < N; ++n) {
        const bool valid = t < seqLengths[n];

        for (int d = 0; d < D; ++d) {
          if (!valid) {
            if (drop_states) {
              H[d] = H_out_qparams.zero_point;
              C[d] = C_out_qparams.zero_point;
            } else {
              H[d] = fbgemm::Requantize<T>(
                  H_prev[d] - H_in_qparams.zero_point, h_in_to_out_params);
              C[d] = fbgemm::Requantize<T>(
                  C_prev[d] - C_in_qparams.zero_point, c_in_to_out_params);
            }
          } else {
            T i_in = fbgemm::Requantize<T>(
                X[d] - X_qparams.zero_point, x_to_sigmoid_params);
            T f_in = fbgemm::Requantize<T>(
                X[1 * D + d] + forget_bias - 2 * X_qparams.zero_point,
                x_to_sigmoid_params);
            T o_in = fbgemm::Requantize<T>(
                X[2 * D + d] - X_qparams.zero_point, x_to_sigmoid_params);
            T g_in = fbgemm::Requantize<T>(
                X[3 * D + d] - X_qparams.zero_point, x_to_tanh_params);

            const T i = sigmoid.Compute(i_in);
            const T f = sigmoid.Compute(f_in);
            const T o = sigmoid.Compute(o_in);
            const T g = tanh.Compute(g_in);
            const T c_prev = C_prev[d];

            // f_times_c_prev.scale = sigmoid_out.scale * c.scale
            int32_t f_times_c_prev = ((int32_t)f - sigmoid_zero_point) *
                ((int32_t)c_prev - C_in_qparams.zero_point);
            // i_times_g.scale = sigmoid_out.scale * tanh_out.scale
            // (higher resolution than f_times_c since often tanh.scale < c.scale)
            int32_t i_times_g =
                ((int32_t)i - sigmoid_zero_point) * ((int32_t)g - tanh_zero_point);

            // c_temp.scale = sigmoid_out.scale * tanh_out.scale
            int32_t f_times_c_prev_rescaled = fbgemm::Requantize<int32_t>(
                f_times_c_prev,
                0,
                c_to_tanh_params.real_multiplier,
                32,
                true /*signed*/);
            int32_t c_temp = f_times_c_prev_rescaled + i_times_g;

            // scale back to c.scale
            C[d] = fbgemm::Requantize<T>(c_temp, c_out_requantization_params);

            T c_tanh_input =
                fbgemm::Requantize<T>(c_temp, c_tanh_requantization_params);
            T host_tanh_c = tanh.Compute(c_tanh_input);

            // o_times_host_tanh_c.scale = sigmoid_out.scale * tanh_out.scale
            int32_t o_times_host_tanh_c = ((int32_t)o - sigmoid_zero_point) *
                ((int32_t)host_tanh_c - tanh_zero_point);
            H[d] =
                fbgemm::Requantize<T>(o_times_host_tanh_c, h_requantization_params);
          }
        }
        H_prev += D;
        C_prev += D;
        X += 4 * D;
        C += D;
        H += D;
      }
    */
}
