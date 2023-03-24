crate::ix!();

pub struct ConcatConverter {
    base: dyn Converter,
}

impl ConcatConverter {

    /**
      | Does not override default converter
      | to OperatorDef
      |
      */
    #[inline] pub fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp =
            std::make_unique<repr::Concat>();
        auto argMap = getArgumentsFromOperator(op);

        auto c = dyn_cast<repr::Concat>(nnOp.get());
        if (argMap.count("axis")) {
          CAFFE_ENFORCE(argMap["axis"].has_i(), "Invalid axis argument");
          int axis = static_cast<int>(argMap["axis"].i());
          c->setAxis(axis);
        }
        if (argMap.count("add_axis")) {
          CAFFE_ENFORCE(argMap["add_axis"].has_i(), "Invalid add_axis argument");
          int add_axis = static_cast<int>(argMap["add_axis"].i());
          c->setAddAxis(!!add_axis);
        }
        return nnOp;
        */
    }
}


