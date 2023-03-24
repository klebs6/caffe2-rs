crate::ix!();

pub struct ConvConverter {
    base: dyn Converter,
}

impl ConvConverter {
    
    /// Does not override default converter to OperatorDef
    #[inline] pub fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp;
        auto argMap = getArgumentsFromOperator(op);
        auto kernelShape = getKernelShape(argMap);
        nnOp = std::make_unique<repr::Conv>(kernelShape);
        auto c = dyn_cast<repr::Conv>(nnOp.get());

        c->setStrides(getStrides(argMap));
        c->setPads(getPads(argMap));
        c->setDilations(getDilations(argMap));
        c->setGroup(getGroup(argMap));

        return nnOp;
        */
    }
}
