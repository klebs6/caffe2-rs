crate::ix!();


pub struct QuantizeDNNLowPOp<T> {

    //USE_OPERATOR_FUNCTIONS(CPUContext);
    base:    OperatorStorage,
    context: CPUContext,

    qfactory:          Box<QuantizationFactory>,
    arguments_parsed:  bool, // default = false
    phantom:           PhantomData<T>,
}

num_inputs!{Quantize, (1,2)}

num_outputs!{Quantize, 1}

identical_type_and_shape_of_input!{Quantize, 0}

register_cpu_operator_with_engine!{
    Quantize,
    DNNLOWP,
    QuantizeDNNLowPOp<u8>
}

register_cpu_operator_with_engine!{
    Quantize,
    DNNLOWP_ROWWISE,
    QuantizeDNNLowPOp<u8>
}

register_cpu_operator_with_engine!{
    Quantize,
    DNNLOWP_16,
    QuantizeDNNLowPOp<u16>
}

register_cpu_operator_with_engine!{
    Quantize,
    DNNLOWP_ROWWISE_16,
    QuantizeDNNLowPOp<u16>
}

register_cpu_operator_with_engine!{
    Int8Quantize,
    DNNLOWP,
    QuantizeDNNLowPOp<u8>
}

register_cpu_operator_with_engine!{
    Int8Quantize,
    DNNLOWP_ROWWISE,
    QuantizeDNNLowPOp<u8>
}

impl<T> QuantizeDNNLowPOp<T> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws),
          qfactory_(dnnlowp::GetQuantizationFactoryOf(this))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
    
        todo!();
        /*
            using namespace dnnlowp;

      if (!arguments_parsed_) {
        dnnlowp::ParseDNNLowPOperatorArguments(this);
        arguments_parsed_ = true;
      }

      CAFFE_ENFORCE(InputSize() <= 2);
      CAFFE_ENFORCE(Input(0).template IsType<float>());

      bool use_input_qparam = false;
      float in_scale = 0;
      int in_zero_point = 0;
      if (InputSize() == 2) {
        use_input_qparam = true;

        const auto* input_qparam_blob =
            Input<caffe2::unique_ptr<Int8QuantParamsBlob>>(1).get();
        CAFFE_ENFORCE(input_qparam_blob);
        in_scale = input_qparam_blob->qparam.scale;
        in_zero_point = input_qparam_blob->qparam.zero_point;
      }

      TensorQuantizationParams in_qparams;

      if (use_input_qparam) {
        in_qparams.scale = in_scale;
        in_qparams.zero_point = in_zero_point;
        in_qparams.precision = qfactory_->GetActivationPrecision();
      } else {
        if (HasStaticQuantization(this)) {
          in_qparams = GetStaticQuantizationParamsOf(this, 0);
        } else {
          in_qparams = GetInputTensorQuantizationParamsOf(this, 0, qfactory_.get());
        }
      }

      int8::Int8TensorCPU* output =
          Outputs()[0]->template GetMutable<int8::Int8TensorCPU>();
      output->t.ResizeLike(Input(0));

      const float* in_data = Input(0).template data<float>();
      T* out_data = output->t.template mutable_data<T>();

      fbgemm::Quantize<T>(in_data, out_data, Input(0).numel(), in_qparams);

      PropagateOutputTensorQuantizationParams(this, 0, in_qparams);

      return true;
        */
    }
}
