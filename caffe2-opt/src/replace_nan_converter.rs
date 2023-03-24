crate::ix!();

pub struct ReplaceNaNConverter { }

impl Converter for ReplaceNaNConverter {
    
    /// Does not override default converter to OperatorDef
    #[inline] fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp =
            std::make_unique<repr::ReplaceNaN>();
        auto argMap = getArgumentsFromOperator(op);

        auto c = dyn_cast<repr::ReplaceNaN>(nnOp.get());
        if (argMap.count("value")) {
          CAFFE_ENFORCE(argMap["value"].has_f(), "Invalid 'value' argument");
          float value = static_cast<float>(argMap["value"].f());
          c->setValue(value);
        }
        return nnOp;
        */
    }
}

register_converter!{ReplaceNaN, ReplaceNaNConverter}
