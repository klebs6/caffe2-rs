crate::ix!();

pub struct DummyAsyncOp {
    storage: OperatorStorage,
    context: CPUContext,
}

register_cpu_operator!{DagUtilTestDummyAsync, DummyAsyncOp}

num_inputs!{DagUtilTestDummyAsync, (0,INT_MAX)}

num_outputs!{DagUtilTestDummyAsync, (0,INT_MAX)}

impl DummyAsyncOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
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
