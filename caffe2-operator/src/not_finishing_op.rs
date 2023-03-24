crate::ix!();

pub struct NotFinishingOp {
    storage: OperatorStorage,
    context: CPUContext,
}

impl NotFinishingOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // never calls SetFinished
        return true;
        */
    }
    
    #[inline] pub fn has_async_part(&self) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
}

register_cpu_operator!{NotFinishingOp, NotFinishingOp}

