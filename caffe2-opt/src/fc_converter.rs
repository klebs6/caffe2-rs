crate::ix!();

pub struct FCConverter {
    base: dyn Converter,
}

impl FCConverter {
    
    /**
      | Does not override default converter
      | to OperatorDef
      |
      */
    #[inline] pub fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp =
            std::make_unique<repr::FC>();
        auto argMap = getArgumentsFromOperator(op);

        auto c = dyn_cast<repr::FC>(nnOp.get());
        if (argMap.count("axis")) {
          CAFFE_ENFORCE(argMap["axis"].has_i(), "Invalid axis argument");
          int axis = static_cast<int>(argMap["axis"].i());
          c->setAxis(axis);
        }
        if (argMap.count("axis_w")) {
          CAFFE_ENFORCE(argMap["axis_w"].has_i(), "Invalid axis_w argument");
          int axis_w = static_cast<int>(argMap["axis_w"].i());
          c->setAxisW(axis_w);
        }

        return nnOp;
        */
    }
}


