#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{align}
x!{const_assert}
x!{core_dispatch}
x!{gradient_maker}
x!{make_signed}
x!{named}
x!{sum_into}
x!{unknown}
x!{util_bench_utils}
x!{util_cast}
x!{util_conversions}
x!{util_cpu_neon}
x!{util_cpuid_test}
x!{util_cpuid}
x!{util_eigen_utils}
x!{util_fatal_signal_asan_no_sig_test}
x!{util_fixed_divisor_test}
x!{util_fixed_divisor}
x!{util_murmur_hash3}
x!{util_signal_handler}
x!{util_string_utils}
x!{util_zmq_helper}
x!{functor}
