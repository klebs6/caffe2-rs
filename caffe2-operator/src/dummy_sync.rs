crate::ix!();

pub struct DummySyncOp {
    storage: OperatorStorage,
    context: CPUContext,
}

register_cpu_operator!{DagUtilTestDummySync, DummySyncOp}

num_inputs!{DagUtilTestDummySync, (0,INT_MAX)}

num_outputs!{DagUtilTestDummySync, (0,INT_MAX)}

impl DummySyncOp {

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
}
