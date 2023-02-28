crate::ix!();

pub struct DequantizeDNNLowPOp<T> {

    //USE_OPERATOR_FUNCTIONS(CPUContext);
    storage:  OperatorStorage,
    context:  CPUContext,
    qfactory: Box<QuantizationFactory>,
    phantom:  PhantomData<T>,
}

num_inputs!{Dequantize, 1}

num_outputs!{Dequantize, 1}

identical_type_and_shape_of_input!{Dequantize, 0}

register_cpu_operator_with_engine!{
    Dequantize,
    DNNLOWP,
    DequantizeDNNLowPOp<u8>
}

register_cpu_operator_with_engine!{
    Dequantize,
    DNNLOWP_ROWWISE,
    DequantizeDNNLowPOp<u8>
}

register_cpu_operator_with_engine!{
    Dequantize,
    DNNLOWP_16,
    DequantizeDNNLowPOp<u16>
}

register_cpu_operator_with_engine!{
    Dequantize,
    DNNLOWP_ROWWISE_16,
    DequantizeDNNLowPOp<u16>
}

register_cpu_operator_with_engine!{
    Int8Dequantize,
    DNNLOWP,
    DequantizeDNNLowPOp<u8>
}

register_cpu_operator_with_engine!{
    Int8Dequantize,
    DNNLOWP_ROWWISE,
    DequantizeDNNLowPOp<u8>
}

register_cpu_operator_with_engine!{
    Int8DequantizeRowWise,
    DNNLOWP,
    DequantizeDNNLowPOp<u8>
}

impl<T> DequantizeDNNLowPOp<T> {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws),
          qfactory_(dnnlowp::GetQuantizationFactoryOf(this)) 

      if (this->debug_def().engine() == "DNNLOWP_16" ||
          this->debug_def().engine() == "DNNLOWP_ROWWISE_16") {
        LOG(WARNING)
            << this->debug_def().engine()
            << " is an experimental feature mostly for testing accuracy with "
               "fixed-point precision higher than 8 and performance is very slow";
      }
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            using namespace dnnlowp;
      TensorQuantizationParams in_qparams =
          GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());

      const TensorCPU& input = InputIsType<int8::Int8TensorCPU>(0)
          ? this->template Input<int8::Int8TensorCPU>(0).t
          : Input(0);

      CAFFE_ENFORCE(input.template IsType<T>());
      Output(0)->ResizeLike(input);
      fbgemm::Dequantize<T>(
          input.template data<T>(),
          Output(0)->template mutable_data<float>(),
          input.numel(),
          in_qparams);

      return true;
        */
    }
}
