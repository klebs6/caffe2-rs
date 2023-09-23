crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/function.h]

pub type Stack        = Vec<IValue>;
pub type Kwargs       = HashMap<String,IValue>;
pub type TaskLauncher = fn(_0: fn() -> ()) -> ();

#[derive(Default,Error)]
pub enum RecursiveMethodCallError {

    #[default]
    Default,
}

pub fn preoptimize_graph(graph: &mut Arc<Graph>)  {
    
    todo!();
        /*
        
        */
}

/**
  | A Function is a pure Graph with no implicit
  | `self` object bound.
  |
  | It contains schema information and the executor
  | that manages the execution of the
  | function. Method is a wrapper around an
  | underlying Function that also provides a `self`
  | object.
  |
  */
pub trait Function:
IsGraphFunction
+ DocString
+ RunMut
+ Run
+ RunAsync
+ Invoke
+ Qualname
+ Name
+ EnsureDefined
+ Graph
+ OptimizedGraph
+ ClearExecutionInfo
+ GetExecutor
+ GetSchema
+ NumInputs
+ CheckSingleOutput
+ PrettyPrintSchema
+ SetSchema 
{
    fn doc_string(&self) -> &String {
        
        todo!();
        /*
            static const string no_doc_string = "";
        return no_doc_string;
        */
    }
}

pub trait IsGraphFunction {
    
    fn is_graph_function(&self) -> bool;
}

pub trait RunMut {

    fn run_mut(&mut self, stack: &mut Stack);
}

pub trait Run {
    
    fn run(&mut self, stack: Stack);
}

pub trait RunAsync {
    
    fn run_async(&mut self, 
        stack:         &mut Stack,
        task_launcher: TaskLauncher) -> IntrusivePtr<Future>;
}

pub trait Invoke {
    
    fn invoke(&mut self, 
        stack:  Vec<IValue>,
        kwargs: &Kwargs) -> IValue;
}

pub trait Qualname {
    
    fn qualname(&self) -> &QualifiedName;
}

pub trait Name {
    
    fn name(&self) -> &String;
}

pub trait EnsureDefined {

    /**
      | if this isn't yet defined, run its method_creator
      | function
      |
      */
    fn ensure_defined(&mut self);
}

pub trait Graph {
    
    fn graph(&self) -> Arc<Graph>;
}

pub trait OptimizedGraph {
    
    fn optimized_graph(&self) -> Arc<Graph>;
}

pub trait ClearExecutionInfo {
    
    fn clear_execution_info(&mut self);
}

pub trait GetExecutor {
    
    fn get_executor(&mut self) -> &mut GraphExecutor;
}

pub trait GetSchema {
    
    fn get_schema(&self) -> &FunctionSchema;
}

pub trait NumInputs {
    
    fn num_inputs(&self) -> usize;
}

pub trait CheckSingleOutput {
    
    fn check_single_output(&mut self);
}

pub trait PrettyPrintSchema {

    
    fn pretty_print_schema(&self) -> String;
}

pub trait SetSchema {
    
    fn set_schema(&mut self, schema: FunctionSchema) -> &mut Function;
}
