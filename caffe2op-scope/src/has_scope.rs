crate::ix!();

/**
  | Checks whether scope blob has any saved
  | scopes left
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct HasScopeOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

caffe_known_type!{WorkspaceStack}

register_cpu_operator!{HasScope, HasScopeOp<CPUContext>}

should_not_do_gradient!{HasScope}

num_inputs!{HasScope, 1}

num_outputs!{HasScope, 1}

impl<Context> HasScopeOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}

impl HasScopeOp<CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
          const auto& ws_stack = OperatorStorage::Input<detail::WorkspaceStack>(0);

          auto* output = Output(0, {1}, at::dtype<bool>());
          bool* output_value = output->template mutable_data<bool>();
          *output_value = !ws_stack.empty();
          return true;
        */
    }
}
