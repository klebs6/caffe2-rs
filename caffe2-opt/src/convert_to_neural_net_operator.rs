crate::ix!();

/**
  | Use these functions instead of the registry
  | directly.
  |
  */
#[inline] pub fn convert_to_neural_net_operator(op: &OperatorDef) -> Box<NeuralNetOperator> {
    
    todo!();
    /*
        auto argMap = Converter::getArgumentsFromOperator(op);

      std::unique_ptr<repr::NeuralNetOperator> nnOp;

      if (ConverterRegistry()->Has(op.type())) {
        nnOp =
            ConverterRegistry()->Create(op.type())->convertToNeuralNetOperator(op);
      }

      if (!nnOp) {
        nnOp = std::make_unique<repr::GenericOperator>(op.type());
      }

      // Generic attributes associated with Ops here
      nnOp->setLayout(getLayout(argMap));

      auto annotation = std::make_unique<Caffe2Annotation>();
      annotation->setOperatorDef(op);

      auto device_name = op.device_option().node_name();
      if (device_name != "") {
        annotation->setDevice(device_name);
      }
      annotation->setDeviceType(op.device_option().device_type());

      nnOp->setAnnotation(std::move(annotation));

      return nnOp;
    */
}

