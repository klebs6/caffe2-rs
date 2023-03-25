crate::ix!();

/**
  | A net test dummy op that does nothing
  | but scaffolding.
  | 
  | Here, we inherit from OperatorStorage
  | because we instantiate on both CPU and
  | 
  | GPU.
  | 
  | In general, you want to only inherit
  | from Operator<Context>.
  |
  */
#[USE_CPU_CONTEXT_OPERATOR_FUNCTIONS]
pub struct NetSimpleRefCountTestOp {
    storage: OperatorStorage,
    context: CPUContext,
}

num_inputs!{NetSimpleRefCountTest, 1}
num_outputs!{NetSimpleRefCountTest, 1}

impl NetSimpleRefCountTestOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const int32_t& input = OperatorStorage::Input<int32_t>(0);
        int32_t* output = OperatorStorage::Output<int32_t>(0);
        *output = input + 1;
        return true;
        */
    }
}
