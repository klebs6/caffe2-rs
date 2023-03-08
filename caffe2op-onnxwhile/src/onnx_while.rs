crate::ix!();

/**
 | EXPERIMENTAL. This operator is
 | a work-in-progress. No assumption should be made
 | about the stability or correctness of this op. ***
 |
 | Generic Looping construct confirming to the ONNX
 | Loop operator spec. This loop has multiple
 | termination conditions:
 |
 | 1. Trip count. Iteration count specified at
 |    runtime. Set by specifying the input
 |    M. Optional. Set to empty string to omit. Note
 |    that a static trip count (specified at graph
 |    construction time) can be specified by passing in
 |    a constant node for input M.
 |
 | 2. Loop termination condition. This is an input to
 |    the op that determines whether to run the first
 |    interation and also a loop-carried dependency for
 |    the body graph. The body graph must yield a value
 |    for the condition variable, whether this input is
 |    provided or not.
 |
 | This table summarizes the operating modes of this
 | operator with equivalent C-style code:
 |
 | Operator inputs defined as (max_trip_count,
 | condition_var). Omitted optional inputs are
 | represented as empty string. Concretely, in this
 | caffe2 op an input is marked as omitted by setting
 | its 'has_{name}' argument to False.
 |
 |     input ("", ""):
 |         for (int i=0; ; ++i) {
 |           cond = ... // Note this value is ignored, but is required in the body
 |         }
 |
 |     input ("", cond) // Note this is analogous to a while loop
 |         bool cond = ...;
 |         for (int i=0; cond; ++i) {
 |           cond = ...;
 |         }
 |
 |     input ("", 1) // Note this is analogous to a do-while loop
 |         bool cond = true
 |         for (int i=0; cond; ++i) {
 |           cond = ...;
 |         }
 |
 |     input (trip_count, "") // Note this is analogous to a for loop
 |         int trip_count = ...
 |         for (int i=0; i < trip_count; ++i) {
 |           cond = ...; // ignored
 |         }
 |
 |     input (trip_count, cond)
 |         int trip_count = ...;
 |         bool cond = ...;
 |         for (int i=0; i < trip_count && cond; ++i) {
 |           cond = ...;
 |         }
 */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ONNXWhileOp<Context> {
    storage:                 OperatorStorage,
    context:                 Context,
    body_net_def:            NetDef,
    parent_ws:               *mut Workspace,
    ws_stack:                WorkspaceStack,
    has_trip_count:          bool,
    has_cond:                bool,
    save_scopes:             bool,
    disable_scopes:          bool,
    num_loop_carried_deps:   i64,
    scope:                   Arc<OnnxWhileOpLocalScope>,
}

num_inputs!{ONNXWhile, (2,INT_MAX)}

num_outputs!{ONNXWhile, (0,INT_MAX)}

inputs!{ONNXWhile, 
    0 => ("max_trip_count",       "Number of iterations to go out to. Used if the flag has_trip_count is True."),
    1 => ("first_iter_condition", "Dynamic condition value for the first iteration. For all subsequent iterations, the condition from the body graph is used. This input is used if the flag has_cond is true.")
}

args!{ONNXWhile, 
    0 => ("body", "Net executed on each iteration"),
    1 => ("has_trip_count", "Whether to use the trip count input"),
    2 => ("has_cond", "Whether to use the condition input"),
    3 => ("save_scopes", "Whether to save the scopes across iterations, as in for backprop"),
    4 => ("disable_scopes", "Do not create new scopes. Use this only if you're certain there will be no name collision, for example if you're converting from a fully-SSA IR")
}

allow_inplace!{ONNXWhile, 
    |input: i32, output: i32| -> bool {
        true
    }
}

register_cpu_operator!{
    ONNXWhile, 
    ONNXWhileOp<CPUContext>
}
