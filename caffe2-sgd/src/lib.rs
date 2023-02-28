#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{sgd_adadelta}
x!{sgd_adagrad_fused}
x!{sgd_adagrad}
x!{sgd_adam}
x!{sgd_clip_tensor}
x!{sgd_fp16_momentum_sgd}
x!{sgd_fp32_momentum_sgd}
x!{sgd_ftrl}
x!{sgd_gftrl}
x!{sgd_iter}
x!{sgd_lars}
x!{sgd_learning_rate_adaption}
x!{sgd_learning_rate_functors}
x!{sgd_learning_rate}
x!{sgd_math_lp}
x!{sgd_momentum_sgd}
x!{sgd_rmsprop}
x!{sgd_rowwise_adagrad_fused}
x!{sgd_rowwise_counter}
x!{sgd_storm}
x!{sgd_weight_scale}
x!{sgd_wngrad}
x!{sgd_yellowfin}
