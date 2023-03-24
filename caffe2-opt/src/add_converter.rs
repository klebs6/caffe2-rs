crate::ix!();

pub struct AddConverter { }

impl Converter for AddConverter {
    
    /**
      | Does not override default converter
      | to OperatorDef
      |
      */
    #[inline] fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp =
            std::make_unique<repr::Add>();
        auto argMap = getArgumentsFromOperator(op);

        auto c = dyn_cast<repr::Add>(nnOp.get());
        if (argMap.count("broadcast")) {
          CAFFE_ENFORCE(argMap["broadcast"].has_i(), "Invalid broadcast argument");
          int broadcast = static_cast<int>(argMap["broadcast"].i());
          c->setBroadcast(!!broadcast);
        }
        return nnOp;
        */
    }
}

register_converter!{Add, AddConverter}

