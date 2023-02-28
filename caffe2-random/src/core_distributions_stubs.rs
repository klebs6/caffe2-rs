crate::ix!();

/**
  | This file provides distributions compatible
  | with ATen/core/DistributionsHelper.h
  | but backed with the std RNG implementation
  | instead of the ATen one.
  | 
  | Caffe2 mobile builds currently do not
  | depend on all of ATen so this is required
  | to allow using the faster ATen RNG for
  | normal builds but keep the build size
  | small on mobile. RNG performance typically
  | doesn't matter on mobile builds since
  | the models are small and rarely using
  | random initialization.
  |
  */
pub type uniform_real_distribution = statrs::distribution::Uniform;
pub type normal_distribution       = statrs::distribution::Normal;
pub type bernouilli_distribution   = statrs::distribution::Bernoulli;
pub type exponential_distribution  = statrs::distribution::Exp;
pub type cauchy_distribution       = statrs::distribution::Cauchy;
pub type lognormal_distribution    = statrs::distribution::LogNormal;
