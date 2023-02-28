crate::ix!();

//default ReluFused == false
pub struct SumDNNLowPOp<T,const ReluFused: bool> {
    //USE_OPERATOR_FUNCTIONS(CPUContext);
    //USE_DNNLOWP_OPERATOR_BASE_FUNCTIONS(T, SumOp<CPUContext>);
    base:                       DNNLowPOp<T, SumOp<CPUContext>>,
    intermediate_qparams:       TensorQuantizationParams,
    out_requantization_params:  RequantizationParams,
}

///---------------------------
pub struct GatherDNNLowPOp<T: PrimInt> {
    //USE_OPERATOR_FUNCTIONS(CPUContext);
    base: GatherOp<CPUContext>,

    fp32_op:                     Box<OpWrapper<GatherOp<CPUContext>,T>>,
    dequantize_output:           bool, // default = false
    measure_quantization_error:  bool, // default = false
    qfactory:                    Box<QuantizationFactory>,
    quantization_error_stats:    QuantizationErrorStats,
    arguments_parsed:            bool, // default = false
    phantom: PhantomData<T>,
}

impl<T: PrimInt> GatherDNNLowPOp<T> {
    
    #[inline] pub fn fp_32op(&mut self) -> *mut OpWrapper<GatherOp<CPUContext>,T> {
        
        todo!();
        /*
            if (!fp32_op_) {
          fp32_op_.reset(
              new OpWrapper<GatherOp<CPUContext>, T>(this, qfactory_.get()));
        }
        return fp32_op_.get();
        */
    }

    #[inline] pub fn do_run_with_type<Index>(&mut self) -> bool {
    
        todo!();
        /*
            // If we endup using it on GPU doing O(N) memcpy is probably not best :)
        // TODO: implement prefetching if it starts mattering (TF does it)
        auto& data = (this->template Input<int8::Int8TensorCPU>(DATA)).t;
        auto& indices = Input(INDICES);
        auto* output = &Outputs()[0]->template GetMutable<int8::Int8TensorCPU>()->t;

        CAFFE_ENFORCE_GE(data.ndim(), 1, "DATA should be at least 1-D");
        auto shape = indices.sizes().vec();
        shape.insert(shape.end(), data.sizes().begin() + 1, data.sizes().end());
        output->Resize(shape);

        int block_size = data.size_from_dim(1);
        auto block_bytesize = data.size_from_dim(1) * data.dtype().itemsize();
        int N = indices.numel();

        auto src_base = static_cast<const char*>(data.raw_data());
        const Index* idxs = indices.template data<Index>();
        auto out = static_cast<char*>(output->raw_mutable_data(data.dtype()));

        for (int i = 0; i < N; ++i) {
          auto idx = idxs[i];
          CAFFE_ENFORCE(
              0 <= idx && idx < data.size(0),
              "INDICES element is out of DATA bounds, id=",
              idx,
              " data_dim=",
              data.size(0));
          auto src = src_base + idx * block_bytesize;
          context_.CopyItemsSameDevice(
              data.dtype(), block_size, src, out + block_bytesize * i);
        }
        return true;
        */
    }

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            using namespace dnnlowp;

      if (!arguments_parsed_) {
        dnnlowp::ParseDNNLowPOperatorArguments(
            this, &dequantize_output_, &measure_quantization_error_);
        arguments_parsed_ = true;
      }

      if (!InputIsType<int8::Int8TensorCPU>(DATA)) {
        if (dequantize_output_) {
          return GatherOp<CPUContext>::RunOnDevice();
        } else {
          // If input or output is float, delegate to fp32 op
          Fp32Op_()->DequantizeInput();
          // dequantize input if it's not already float
          if (!Fp32Op_()->Get()->RunOnDevice()) {
            return false;
          }

          int8::Int8TensorCPU* output =
              Outputs()[0]->template GetMutable<int8::Int8TensorCPU>();

          output->t.ResizeLike(*Fp32Op_()->Get()->Output(0));
          T* out_data = output->t.template mutable_data<T>();

          TensorQuantizationParams out_qparams;
          if (HasStaticQuantization(this)) {
            out_qparams = GetStaticQuantizationParamsOf(this, 0);
          } else {
            out_qparams = Fp32Op_()->GetOutputQuantizationParams(qfactory_.get());
          }

          fbgemm::Quantize<T>(
              static_cast<const float*>(Fp32Op_()->Get()->Output(0)->raw_data()),
              out_data,
              output->t.numel(),
              out_qparams);

          PropagateOutputTensorQuantizationParams(this, 0, out_qparams);
        }
      } else {
        DispatchHelper<TensorTypes<int32_t, int64_t>>::call(this, Input(INDICES));

        TensorQuantizationParams in_qparams =
            GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());

        PropagateOutputTensorQuantizationParams(this, 0, in_qparams);
      }

      return true;
        */
    }
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : GatherOp<CPUContext>(operator_def, ws),
          qfactory_(dnnlowp::GetQuantizationFactoryOf(this))
        */
    }
}

#[inline] pub fn element_wise_sumavx2<T, const ReluFused: bool>(
    input0:        *const T,
    input1:        *const T,
    output:        *mut T,
    len:           i32,
    a_scale:       f32,
    a_zero_point:  i32,
    b_scale:       f32,
    b_zero_point:  i32,
    c_scale:       f32,
    c_zero_points: i32)  {

    todo!();
    /*
    
    */
}

impl<T: PrimInt> Drop for GatherDNNLowPOp<T> {
    fn drop(&mut self) {
        todo!();
        /* 
      if (measure_quantization_error_) {
        dnnlowp::ReportQuantizationError(this, quantization_error_stats_);
      }
 */
    }
}

register_cpu_operator_with_engine!{
    Gather, 
    DNNLOWP, 
    GatherDNNLowPOp<u8>
}

register_cpu_operator_with_engine!{
    Int8Gather,
    DNNLOWP,
    GatherDNNLowPOp<u8>
}
