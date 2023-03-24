crate::ix!();

/**
  | AsyncTaskGraph represents an execution
  | of a net, it owns the tasks and associated
  | futures, sets up future callbacks and
  | propagates errors.
  | 
  | Usage steps:
  | 
  | - Adding graph nodes and edges through
  | CreateNode/AddDependency;
  | 
  | - Freezing the graph (FreezeGraph),
  | after the freezing a future can be obtained
  | using GetFuture;
  | 
  | - Execution of the graph is scheduled
  | through ExecuteGraph, after each execution
  | Reset must be called to prepare the graph
  | for the next run
  |
  */
pub trait AsyncTaskGraphBase {

    fn create_node(&mut self, node_id: i32, ops: &Vec<*mut OperatorStorage>) -> bool;

    fn add_dependency(&mut self, child_node_id: i32, parent_node_ids: &Vec<i32>) -> bool;

    fn freeze_graph(&mut self);

    fn execute_graph(&mut self) -> *mut AsyncTaskFuture;

    fn get_future(&mut self) -> *mut AsyncTaskFuture;

    fn reset(&mut self);
}
