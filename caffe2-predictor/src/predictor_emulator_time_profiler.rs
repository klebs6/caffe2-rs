crate::ix!();

pub mod emulator {

    use crate::{
        Profiler
    };

    /**
      | A profiler that measures the walltime
      | of a @runnable
      |
      */
    pub struct TimeProfiler { }

    impl Profiler for TimeProfiler {

        fn profile(&self, runnable: fn() -> ()) -> f32 {
            todo!();

            /*
            Timer timer;
            runnable();
            return timer.MilliSeconds();
            */
        }
    }
}
