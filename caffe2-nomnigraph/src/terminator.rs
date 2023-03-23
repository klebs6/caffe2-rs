crate::ix!();

pub struct Terminator {
    base: Instruction,
}

impl Terminator {

    pub fn new(op: Opcode) -> Self {
    
        todo!();
        /*
            : Instruction(op)
        */
    }
    
    #[inline] pub fn classof(v: *const Value) -> bool {
        
        todo!();
        /*
            return isa<Instruction>(V) &&
                    isTerminator(cast<Instruction>(V)->getOpcode());
        */
    }
    
    #[inline] pub fn is_terminator(op: &Opcode) -> bool {
        
        todo!();
        /*
            return op >= Opcode::TerminatorStart && op <= Opcode::TerminatorEnd;
        */
    }
}
