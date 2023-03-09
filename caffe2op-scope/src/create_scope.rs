crate::ix!();

/**
  | 'CreateScope' operator initializes
  | and outputs empty scope that is used
  | by Do operator to store local blobs
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct CreateScopeOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

register_cpu_operator!{CreateScope, CreateScopeOp<CPUContext>}

should_not_do_gradient!{CreateScope}

num_inputs!{CreateScope, 0}

num_outputs!{CreateScope, 1}

impl<Context> CreateScopeOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}

impl CreateScopeOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto* ws_stack = OperatorStorage::Output<detail::WorkspaceStack>(0);
      ws_stack->clear();
      return true;
        */
    }
}
