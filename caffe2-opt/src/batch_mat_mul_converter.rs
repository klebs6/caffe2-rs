crate::ix!();

pub struct BatchMatMulConverter {
    base: dyn Converter,
}

register_converter!{BatchMatMul, BatchMatMulConverter}

trivial_converter!{BatchGather}

register_converter!{BatchGather, BatchGatherConverter}

impl BatchMatMulConverter {

    /**
      | Does not override default converter
      | to OperatorDef
      |
      */
    #[inline] pub fn convert_to_neural_net_operator(&mut self,
        op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp =
            std::make_unique<repr::BatchMatMul>();
        auto argMap = getArgumentsFromOperator(op);

        auto c = dyn_cast<repr::BatchMatMul>(nnOp.get());
        if (argMap.count("trans_a")) {
          CAFFE_ENFORCE(argMap["trans_a"].has_i(), "Invalid axis argument");
          int trans_a = static_cast<int>(argMap["trans_a"].i());
          c->setTransA(!!trans_a);
        }
        if (argMap.count("trans_b")) {
          CAFFE_ENFORCE(argMap["trans_b"].has_i(), "Invalid add_axis argument");
          int trans_b = static_cast<int>(argMap["trans_b"].i());
          c->setTransB(!!trans_b);
        }
        if (argMap.count("broadcast")) {
          CAFFE_ENFORCE(argMap["broadcast"].has_i(), "Invalid add_axis argument");
          int broadcast = static_cast<int>(argMap["broadcast"].i());
          c->setBroadcast(!!broadcast);
        }
        return nnOp;
        */
    }
}

