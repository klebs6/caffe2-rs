crate::ix!();

pub struct SyncErrorOp {
    storage: OperatorStorage,
    context: CPUContext,
    fail:    bool,
    throw:   bool,
}

impl SyncErrorOp {

    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : Operator<CPUContext>(operator_def, ws),
            fail_(OperatorStorage::GetSingleArgument<bool>("fail", true)),
            throw_(OperatorStorage::GetSingleArgument<bool>("throw", false))
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            if (fail_) {
          if (throw_) {
            throw std::logic_error("Error");
          } else {
            return false;
          }
        } else {
          return true;
        }
        */
    }
}

register_cpu_operator!{SyncErrorOp, SyncErrorOp}

