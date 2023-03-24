crate::ix!();

pub struct ClipConverter {
    base: dyn Converter,
}

impl ClipConverter {
    
    /// Does not override default converter to OperatorDef
    #[inline] pub fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            auto argMap = getArgumentsFromOperator(op);
        float min = std::numeric_limits<float>::lowest();
        float max = float::max;

        if (argMap.count("min")) {
          CAFFE_ENFORCE(argMap["min"].has_f(), "Invalid 'min' argument");
          min = static_cast<float>(argMap["min"].f());
        }

        if (argMap.count("max")) {
          CAFFE_ENFORCE(argMap["max"].has_f(), "Invalid 'max' argument");
          max = static_cast<float>(argMap["max"].f());
        }

        return std::make_unique<repr::Clip>(min, max);
        */
    }
}


