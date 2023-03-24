crate::ix!();

pub struct TracingConfig {
    mode:                      TracingMode, // {TracingMode::EVERY_K_ITERATIONS};
    filepath:                  String,      // {"/tmp"};

    /// for TracingMode::EVERY_K_ITERATIONS
    trace_every_nth_batch:     i64, // default = 100
    dump_every_nth_batch:      i64, // default = 10000

    // for TracingMode::GLOBAL_TIMESLICE
    trace_every_n_ms:          i64, // = 2 * 60 * 1000; // 2min
    trace_for_n_ms:            i64, // default = 1000 // 1sec
}
