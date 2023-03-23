crate::ix!();

///-----------------------------------------------
/// \brief All the different types of execution.
pub enum Opcode {

    /// Handles basic instructions.
    Generic,          

    /// LLVM style range of operations.
    TerminatorStart, 

    Branch,
    Return,
    TerminatorEnd,
    Phi,
}

pub struct Instruction {
    base: Value,
    op:   Opcode,
}

impl Default for Instruction {
    
    fn default() -> Self {
        todo!();
        /*
            : Value(ValueKind::Instruction), op_(Opcode::Generic
        */
    }
}

impl Instruction {
    
    pub fn new(op: Opcode) -> Self {
    
        todo!();
        /*
            : Value(ValueKind::Instruction), op_(op)
        */
    }
    
    #[inline] pub fn classof(v: *const Value) -> bool {
        
        todo!();
        /*
            return V->getKind() == ValueKind::Instruction;
        */
    }
    
    #[inline] pub fn get_opcode(&self) -> Opcode {
        
        todo!();
        /*
            return op_;
        */
    }
}

