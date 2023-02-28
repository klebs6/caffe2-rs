crate::ix!();

pub enum ValueKind { Value, Instruction, Data }

pub struct Value {

    kind: ValueKind,
}

impl Default for Value {
    
    fn default() -> Self {
        todo!();
        /*
            : kind_(ValueKind::Value
        */
    }
}

impl Value {
    
    pub fn new(k: ValueKind) -> Self {
    
        todo!();
        /*
            : kind_(K)
        */
    }
    
    #[inline] pub fn get_kind(&self)  {
        
        todo!();
        /*
            return kind_;
        */
    }
}

///-----------------------------------------------
pub struct Data {
    base:    Value,
    version: usize, // default = 0
}

impl Default for Data {
    
    fn default() -> Self {
        todo!();
        /*
            : Value(ValueKind::Data
        */
    }
}

impl Data {
    
    #[inline] pub fn classof(v: *const Value) -> bool {
        
        todo!();
        /*
            return V->getKind() == ValueKind::Data;
        */
    }
    
    #[inline] pub fn get_version(&self) -> usize {
        
        todo!();
        /*
            return version_;
        */
    }
    
    #[inline] pub fn set_version(&mut self, version: usize)  {
        
        todo!();
        /*
            version_ = version;
        */
    }
}

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


///--------------------------------

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

///--------------------------------

pub struct Branch {
    base: Terminator,
}

impl Default for Branch {
    
    fn default() -> Self {
        todo!();
        /*
            : Terminator(Instruction::Opcode::Branch
        */
    }
}

///--------------------------------

pub struct Return {
    base: Terminator,
}

impl Default for Return {
    
    fn default() -> Self {
        todo!();
        /*
            : Terminator(Instruction::Opcode::Return
        */
    }
}

///--------------------------------

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
