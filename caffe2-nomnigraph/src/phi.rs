crate::ix!();

pub struct Phi {
    base: Instruction,
}

impl Default for Phi {
    
    fn default() -> Self {
        todo!();
        /*
            : Instruction(Instruction::Opcode::Phi
        */
    }
}

pub struct NNPhi {
    //NOMNIGRAPH_DEFINE_NN_RTTI(NNPhi);
    base: NeuralNetOperator,
}

impl Default for NNPhi {
    
    fn default() -> Self {
        todo!();
        /*
            : NeuralNetOperator(NNKind::NNPhi, Opcode::Phi
        */
    }
}

