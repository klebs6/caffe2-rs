crate::ix!();

pub struct OperatorNode {
    operator:              Box<OperatorStorage>,
    children:              Vec<i32>,
    parents:               Vec<i32>,
    runtime_parent_count:  Atomic<i32>,
    is_chain_start:        bool, // default = false
    scheduled:             AtomicBool, // default = ATOMIC_FLAG_INIT
}

pub struct OpGraphNode {
    children:          Vec<i32>,
    parents:           Vec<i32>,
    visited_inputs:    i32, // default = 0
    num_orig_parents:  i32,
}

pub type ExecutionChains = HashMap<i32, Vec<i32>>;
