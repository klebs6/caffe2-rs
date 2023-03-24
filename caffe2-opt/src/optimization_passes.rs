crate::ix!();

/**
  | This file sets up the optimization pass
  | registry.
  | 
  | You'll want to either create a class
  | that inherits from OptimizationPass
  | and implements run or use the
  | 
  | REGISTER_OPT_PASS_FROM_FUNC(name,
  | func) to register a function that takes
  | in an NNModule*.
  | 
  | If you need access to the workspace in
  | the optimization you'll need to use
  | a different registry and inherit from
  | 
  | WorkspaceOptimizationPass.
  |
  */
pub struct OptimizationPass<T,U> {
    nn:  *mut NNModule<T,U>,
}

pub trait OptimizationPassTrait {
    fn run();
}

impl<T,U> OptimizationPass<T,U> {
    
    pub fn new(nn: *mut NNModule<T,U>) -> Self {
    
        todo!();
        /*
            : nn_(nn)
        */
    }
}

pub struct WorkspaceOptimizationPass<T,U> {
    base: OptimizationPass<T,U>,

    ws:  *mut Workspace,
}

impl<T,U> WorkspaceOptimizationPass<T,U> {
    
    pub fn new(nn: *mut NNModule<T,U>, ws: *mut Workspace) -> Self {
    
        todo!();
        /*
            : OptimizationPass(nn), ws_(ws)
        */
    }
}

declare_registry!{
    WorkspaceOptimizationPassRegistry,
    WorkspaceOptimizationPass,
    *mut NNModule,
    *mut Workspace
}


declare_registry!{
    OptimizationPassRegistry, 
    OptimizationPass, 
    NNModule
}


define_registry!{
    /*
    WorkspaceOptimizationPassRegistry,
    WorkspaceOptimizationPass,
    NNModule*,
    Workspace*
    */
}

define_registry!{
    /*
    OptimizationPassRegistry, 
    OptimizationPass, 
    NNModule*
    */
}
