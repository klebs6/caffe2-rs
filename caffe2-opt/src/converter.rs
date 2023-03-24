crate::ix!();

define_registry!{/*ConverterRegistry, Converter*/}

pub trait Converter {

    fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator>;

    #[inline] fn get_arguments_from_operator(&self, op: OperatorDef) -> HashMap<String,Argument> {
        
        todo!();
        /*
            std::map<std::string, caffe2::Argument> argMap;
      for (auto arg : op.arg()) {
        argMap[arg.name()] = arg;
      }
      return argMap;
        */
    }

    #[inline] fn convert_to_operator_def(&mut self, nn_op: *const NeuralNetOperator) -> OperatorDef {
        
        todo!();
        /*
            auto* annotation = nnOp->getAnnotation();
      // Default to using the stored operator.
      if (annotation && isa<Caffe2Annotation>(annotation)) {
        return dyn_cast<Caffe2Annotation>(annotation)->getOperatorDef();
      }
      LOG(WARNING)
          << "Cannot instantiate this OperatorDef from nomnigraph, falling back";
      caffe2::OperatorDef op;
      op.set_type(nnOp->getName());
      return op;
        */
    }
    
    #[inline] fn get_device_option(&self, nn_op: *const NeuralNetOperator) -> DeviceOption {
        
        todo!();
        /*
            auto* annotation = nnOp->getAnnotation();
      // Default to using the stored operator.
      if (annotation && isa<Caffe2Annotation>(annotation)) {
        return dyn_cast<Caffe2Annotation>(annotation)
            ->getOperatorDef()
            .device_option();
      }
      caffe2::DeviceOption opt;
      return opt;
        */
    }
    
    #[inline] fn get_kernel_shape(&mut self, arg_map: HashMap<String,Argument>) -> Vec<i32> {
        
        todo!();
        /*
            // There are literally three ways to define shapes in Conv in Caffe2
      std::vector<int> kernelShape;
      if (argMap.count("kernel")) {
        CAFFE_ENFORCE(argMap["kernel"].has_i(), "Invalid kernel argument");
        int kernel = static_cast<int>(argMap["kernel"].i());
        kernelShape = {kernel, kernel};
      } else if (argMap.count("kernels")) {
        for (auto i : argMap["kernels"].ints()) {
          kernelShape.push_back(static_cast<int>(i));
        }
      } else if (argMap.count("kernel_h") && argMap.count("kernel_w")) {
        CAFFE_ENFORCE(argMap["kernel_h"].has_i(), "Invalid kernel argument");
        CAFFE_ENFORCE(argMap["kernel_w"].has_i(), "Invalid kernel argument");
        int kernelH = static_cast<int>(argMap["kernel_h"].i());
        int kernelW = static_cast<int>(argMap["kernel_w"].i());
        kernelShape = {kernelH, kernelW};
      }
      return kernelShape;
        */
    }
}

declare_registry!{
    ConverterRegistry, 
    Converter
}
