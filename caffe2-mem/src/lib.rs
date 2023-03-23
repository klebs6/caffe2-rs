#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{compute_blob_recycling_for_dag}
x!{optimize_inference_net}
x!{run_schema_check}
