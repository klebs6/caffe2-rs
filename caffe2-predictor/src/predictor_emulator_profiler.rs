crate::ix!();

/**
  | An interface to profile the metrics
  | of a @runnable.
  | 
  | It should return execution walltime
  | in milliseconds.
  |
  */
pub trait Profiler {

    fn profile(&self, runnable: fn() -> ()) -> f32;
}
