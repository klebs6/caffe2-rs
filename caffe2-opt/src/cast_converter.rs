crate::ix!();

pub struct CastConverter { }

impl Converter for CastConverter {
    
    /// Does not override default converter to OperatorDef
    #[inline] fn convert_to_neural_net_operator(&mut self, op: &OperatorDef) -> Box<NeuralNetOperator> {
        
        todo!();
        /*
            std::unique_ptr<repr::NeuralNetOperator> nnOp =
            std::make_unique<repr::Cast>();
        auto argMap = getArgumentsFromOperator(op);

        auto c = dyn_cast<repr::Cast>(nnOp.get());
        ArgumentHelper helper(op);
        c->setTo(cast::GetCastDataType(helper, "to"));
        return nnOp;
        */
    }
}

register_converter!{Cast, CastConverter}
