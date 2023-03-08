/*!
  | Note(Yangqing): I think it is possible to do
  | a more general swapaxes operator but I am
  | a little afraid of going down that general
  | path. Only implementing the two actually needed
  | ones here.
  */

#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{cudnn_order_switch}
x!{get_gradient}
x!{nchw2nhwc}
x!{nchw2nhwc_cudnn}
x!{nhwc2nchw}
x!{nhwc2nchw_cudnn}
