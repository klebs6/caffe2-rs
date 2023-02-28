#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

#[macro_export] macro_rules! x { 
    ($x:ident) => { 
        mod $x; 
        pub use $x::*; 
    }
}

#[macro_export] macro_rules! ix { 
    () => { 
        use crate::{ 
            imports::* , 
        };
        use crate::*;
    } 
}

x!{caffe2_legacy}
x!{caffe2}
x!{hsm}
x!{metanet}
x!{predictor_consts}
x!{prof_dag}
x!{torch}
x!{util_proto_utils_test}
x!{util_proto_utils}
x!{util_proto_wrap}
