crate::ix!();

pub struct SumRelu {
    //NOMNIGRAPH_DEFINE_NN_RTTI(SumRelu);
    base: NeuralNetOperator,
}

impl Default for SumRelu {
    
    fn default() -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::SumRelu
        */
    }
}

impl From<&Sum> for SumRelu {
    
    fn from(sum: &Sum) -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::SumRelu)
        */
    }
}

