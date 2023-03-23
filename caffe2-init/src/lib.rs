#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{check_intrinsics}
x!{global_init}
x!{global_init_is_called_guard}
x!{global_init_state}
x!{init_registerer}
x!{initialize_registry}
x!{open_mpthreads}
x!{quit_if_feature_unsupported}
x!{register}
x!{set_denormals}
x!{set_mklthreads}
x!{test_init}
x!{unsafe_run}
x!{warn_if_feature_unused}
