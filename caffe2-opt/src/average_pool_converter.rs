crate::ix!();

pub struct AveragePoolConverter {
    base: dyn Converter,
}

impl AveragePoolConverter {
    
    /// Does not override default converter to OperatorDef
    #[inline] pub fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp;
        auto argMap = getArgumentsFromOperator(op);
        auto kernelShape = getKernelShape(argMap);
        nnOp = std::make_unique<repr::AveragePool>(kernelShape);
        return nnOp;
        */
    }
}
