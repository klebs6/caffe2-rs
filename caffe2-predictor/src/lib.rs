#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{predictor_config}
x!{predictor_emulator_benchmark}
x!{predictor_emulator_data_filler_test}
x!{predictor_emulator_data_filler}
x!{predictor_emulator_emulator}
x!{predictor_emulator_net_supplier}
x!{predictor_emulator_output_formatter}
x!{predictor_emulator_profiler}
x!{predictor_emulator_std_output_formatter}
x!{predictor_emulator_time_profiler}
x!{predictor_emulator_utils}
x!{predictor_inferencegraph}
x!{predictor_predictor}
x!{predictor_test}
x!{predictor_threadlocalptr}
x!{predictor_transforms}
x!{predictor_utils}
