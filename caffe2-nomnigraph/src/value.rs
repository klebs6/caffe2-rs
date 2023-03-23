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

