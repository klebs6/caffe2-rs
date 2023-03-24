crate::ix!();

#[inline] pub fn start_iter(tracer: &Arc<Tracer>) -> bool {
    
    todo!();
    /*
        if (!tracer) {
        return false;
      }
      auto iter = tracer->bumpIter();
      bool is_enabled;
      bool should_dump;
      if (tracer->config().mode == TracingMode::EVERY_K_ITERATIONS) {
        is_enabled = iter % tracer->config().trace_every_nth_batch == 0;
        should_dump = iter % tracer->config().dump_every_nth_batch == 0;
      } else {
        using namespace std::chrono;
        auto ms =
            duration_cast<milliseconds>(system_clock::now().time_since_epoch())
                .count();
        is_enabled = (ms % tracer->config().trace_every_n_ms) <
            tracer->config().trace_for_n_ms;
        // dump just after disabled tracing
        should_dump = tracer->isEnabled() && !is_enabled;
      }
      tracer->setEnabled(is_enabled);
      if (should_dump) {
        int dumping_iter = tracer->bumpDumpingIter();
        tracer->dumpTracingResultAndClearEvents(c10::to_string(dumping_iter));
      }
      return is_enabled;
    */
}

