crate::ix!();

#[cfg(openmp)]
declare_int!{caffe2_omp_num_threads}

declare_bool!{caffe2_dnnlowp_shared_int32_buffer}
declare_bool!{caffe2_force_shared_col_buffer}

// TODO: code duplication with dnnlowp_op.h
pub struct ConvPoolDNNLowPOpBase<T: PrimInt,FP32_OP> {
    //USE_CONV_POOL_BASE_FUNCTIONS(CPUContext);
    base: ConvPoolOpBase<CPUContext>,

    measure_quantization_error:  bool, // default = false
    followed_by:                 String,
    in_qparams:                  Vec<TensorQuantizationParams>,
    out_qparams:                 TensorQuantizationParams,
    fp32_op:                     Box<OpWrapper<FP32_OP,T>>,
    qfactory:                    Box<QuantizationFactory>,
    out_temp:                    Vec<T>,

    /**
      | Buffer to store quantized output temporarily
      | when we output dequantized values.
      |
      */
    quantization_error_stats:    QuantizationErrorStats,

    arguments_parsed:            bool, // default = false
}

impl<T: PrimInt, FP32_OP> Drop for ConvPoolDNNLowPOpBase<T,FP32_OP> {
    fn drop(&mut self) {
        todo!();
        /* 
        if (measure_quantization_error_) {
          dnnlowp::ReportQuantizationError(this, quantization_error_stats_);
          LOG(WARNING) << this->debug_def().output(0) << " with type "
                       << this->debug_def().type() << " has output qparams : "
                       << "scale " << out_qparams_.scale << " offset "
                       << out_qparams_.zero_point << "; ";
        }
       */
    }
}

impl<T: PrimInt, FP32_OP> ConvPoolDNNLowPOpBase<T, FP32_OP> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : ConvPoolOpBase<CPUContext>(operator_def, ws),
            in_qparams_(InputSize()),
            qfactory_(dnnlowp::GetQuantizationFactoryOf(this)) 

    #ifdef _OPENMP
        if (FLAGS_caffe2_omp_num_threads > 0) {
          omp_set_num_threads(FLAGS_caffe2_omp_num_threads);
        }
    #endif

        if (this->debug_def().engine() == "DNNLOWP_16" ||
            this->debug_def().engine() == "DNNLOWP_ROWWISE_16") {
          LOG(WARNING)
              << this->debug_def().engine()
              << " is an experimental feature mostly for testing accuracy with "
                 "fixed-point precision higher than 8 and performance is very slow";
        }
        */
    }
    
    #[inline] pub fn input_tensorcpu(&mut self, idx: i32) -> &TensorCPU {
        
        todo!();
        /*
            if (InputIsType<int8::Int8TensorCPU>(idx)) {
          return this->Input<int8::Int8TensorCPU>(idx).t;
        } else if (InputIsType<Int8ConvDNNLowPPackedWeightBlob>(idx)) {
          return this->Input<Int8ConvDNNLowPPackedWeightBlob>(idx).original_tensor;
        } else {
          return Input(idx);
        }
        */
    }
    
    #[inline] pub fn output_tensorcpu_with_idx(&mut self, idx: i32) -> *mut TensorCPU {
        
        todo!();
        /*
            return &Outputs()[idx]->template GetMutable<int8::Int8TensorCPU>()->t;
        */
    }
    
    #[inline] pub fn output_tensorcpu_with_idx_dims_and_options(&mut self, 
        idx:     i32,
        dims:    &[i32],
        options: TensorOptions) -> *mut Tensor {
        
        todo!();
        /*
            auto* t = &Outputs()[idx]->template GetMutable<int8::Int8TensorCPU>()->t;
        ReinitializeTensor(t, dims, options.device(CPU));
        return t;
        */
    }
    
    #[inline] pub fn get_quantized_output_data(&mut self) -> *mut T {
        
        todo!();
        /*
            return OutputTensorCPU_(0)->template mutable_data<T>();
        */
    }
    
    #[inline] pub fn measure_quantization_error(&mut self)  {
        
        todo!();
        /*
            if (!measure_quantization_error_ || !Fp32Op_()) {
          return;
        }

        const float* actual = nullptr;
        vector<float> actual_temp;
        if (OutputTensorCPU_(0)->template IsType<float>()) {
          actual = OutputTensorCPU_(0)->template data<float>();
        } else {
          actual_temp.resize(OutputTensorCPU_(0)->numel());
          fbgemm::Dequantize<T>(
              OutputTensorCPU_(0)->template data<T>(),
              actual_temp.data(),
              OutputTensorCPU_(0)->numel(),
              out_qparams_);
          actual = actual_temp.data();
        }

        TensorCPU* float_tensor = Fp32Op_()->Get()->Output(0);
        float* ref = float_tensor->template mutable_data<float>();
        if (followed_by_ == "Relu" || debug_def().type() == "ConvRelu" ||
            debug_def().type() == "Int8ConvRelu") {
          for (int i = 0; i < OutputTensorCPU_(0)->numel(); ++i) {
            ref[i] = std::max(0.f, ref[i]);
          }
        }

        dnnlowp::MeasureQuantizationError(
            actual, ref, OutputTensorCPU_(0)->numel(), &quantization_error_stats_);
        */
    }
    
    #[inline] pub fn run_on_device_epilogue(&mut self)  {
        
        todo!();
        /*
            dnnlowp::PropagateOutputTensorQuantizationParams(this, 0, out_qparams_);

        MeasureQuantizationError_();
        */
    }
    
    #[inline] pub fn parse_dnn_lowp_operator_arguments(&mut self)  {
        
        todo!();
        /*
            if (!arguments_parsed_) {
          bool dequantize_output;
          dnnlowp::ParseDNNLowPOperatorArguments(
              this,
              &dequantize_output,
              &measure_quantization_error_,
              &followed_by_);
          CAFFE_ENFORCE_EQ(
              dequantize_output,
              false,
              "Conv DNNLOWP operators don't support dequantize_output");
          arguments_parsed_ = true;
        }
        */
    }
    
    #[inline] pub fn get_output_quantization_params(&mut self)  {
        
        todo!();
        /*
            using namespace dnnlowp;

        ParseDNNLowPOperatorArguments_();

        if (HasStaticQuantization(this)) {
          out_qparams_ = GetStaticQuantizationParamsOf(this, 0);

          if (measure_quantization_error_) {
            // To measure quantization error, run ref fp32 impl.
            // This doesn't really belong here but we need to run the reference fp32
            // implementation before quantized computation of some inplace operators
            // will overwrite their inputs.
            Fp32Op_()->DequantizeInput();
            Fp32Op_()->Get()->RunOnDevice();
          }
        } else {
          // TODO: this is only needed when dequantize_output_ == false but leave
          // as it is now because some code relies on out_qparams_ initialized even
          // though it never actually uses it.
          Fp32Op_()->DequantizeInput();
          Fp32Op_()->Get()->RunOnDevice();
          out_qparams_ = Fp32Op_()->GetOutputQuantizationParams(qfactory_.get());
        }
        */
    }
    
    #[inline] pub fn fp_32op(&mut self) -> *mut OpWrapper<FP32_OP,T> {
        
        todo!();
        /*
            if (!fp32_op_) {
          fp32_op_.reset(new OpWrapper<FP32_OP, T>(this, qfactory_.get()));
        }
        return fp32_op_.get();
        */
    }
    
    #[inline] pub fn create_shared_int_32buffer(&mut self)  {
        
        todo!();
        /*
            auto* mutexPtr =
            ws_->CreateBlob("__CAFFE2_DNNLOWP_SHARED_INT32_BUFFER_CPU_MUTEX__")
                ->GetMutable<std::unique_ptr<std::mutex>>();
        mutexPtr->reset(new std::mutex());
        ws_->CreateBlob("__CAFFE2_DNNLOWP_SHARED_INT32_BUFFER_CPU__");
        */
    }

    pub fn run_with_shared_buffer(&mut self, 
        col_buffer: *mut Tensor, 
        y_int32:    *mut Vec<i32>, 
        f:          fn(col_buffer_shared: *mut Tensor, y_int32_shared: *mut Vec<i32>) -> ()) {

        todo!();

        /*
        auto f2 = [this, Y_int32, f](Tensor* col_buffer_shared) {
          if (FLAGS_caffe2_dnnlowp_shared_int32_buffer) {
            auto* mutexBlob =
                ws_->GetBlob("__CAFFE2_DNNLOWP_SHARED_INT32_BUFFER_CPU_MUTEX__");
            CAFFE_ENFORCE(mutexBlob, "Must call CreateSharedInt32Buffer() first");

            auto* mutexPtr = mutexBlob->GetMutable<std::unique_ptr<std::mutex>>();
            std::lock_guard<std::mutex> g(**mutexPtr);

            auto* Y_int32_shared =
                ws_->GetBlob("__CAFFE2_DNNLOWP_SHARED_INT32_BUFFER_CPU__")
                    ->template GetMutable<vector<int32_t>>();
            f(col_buffer_shared, Y_int32_shared);
          } else {
            f(col_buffer_shared, Y_int32);
          }
        };

        if (FLAGS_caffe2_force_shared_col_buffer || this->shared_buffer_) {
          runWithSharedBuffer<CPUContext>(this->ws_, f2);
        } else {
          f2(col_buffer);
        }
        */
    }
}

#[macro_export] macro_rules! use_conv_pool_dnnlowp_operator_base_functions {
    ($T:ident, $FP32_OP:ident) => {
        todo!();
        /*
        
          /* using override */ using BaseType = ConvPoolDNNLowPOpBase<T, FP32_OP>; 
          /* using override */ using BaseType::GetOutputQuantizationParams_;       
          /* using override */ using BaseType::GetQuantizedOutputData_;            
          /* using override */ using BaseType::Fp32Op_;                            
          /* using override */ using BaseType::InputTensorCPU_;                    
          /* using override */ using BaseType::MeasureQuantizationError_;          
          /* using override */ using BaseType::OutputTensorCPU_;                   
          /* using override */ using BaseType::RunOnDeviceEpilogue_;               
          /* using override */ using BaseType::followed_by_;                       
          /* using override */ using BaseType::in_qparams_;                        
          /* using override */ using BaseType::measure_quantization_error_;        
          /* using override */ using BaseType::out_qparams_;                       
          /* using override */ using BaseType::qfactory_;
        */
    }
}
