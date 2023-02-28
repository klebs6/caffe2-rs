#![feature(trait_alias)]

#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{core_net_async_base}
x!{core_net_async_scheduling}
x!{core_net_async_task_future}
x!{core_net_async_task_graph}
x!{core_net_async_task}
x!{core_net_async_tracing_test}
x!{core_net_async_tracing}
x!{core_net_dag_utils_test}
x!{core_net_dag_utils}
x!{core_net_gpu_test}
x!{core_net_parallel}
x!{core_net_simple_refcount_test}
x!{core_net_simple_refcount}
x!{core_net_simple}
x!{core_net_test}
x!{core_net}
x!{core_observer_test}
x!{core_observer}
x!{core_operator_gpu_test}
x!{core_operator_gradient}
x!{core_operator_schema_test}
x!{core_operator_schema}
x!{core_operator_test}
x!{core_operator}
x!{core_parallel_net_test}
x!{core_workspace_test}
x!{core_workspace}
x!{observer_operator_attaching_net_observer}
x!{observer_profile_observer}
x!{observer_runcnt_observer}
x!{observer_time_observer_test}
x!{observer_time_observer}
