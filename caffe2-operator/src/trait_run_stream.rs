crate::ix!();

pub trait RunStream {
 
    /**
      | The run function of Operator switches to
      | the device, and then carries out the
      | actual computation with RunOnDevice(). You
      | should implement RunOnDevice instead of
      | Run().
      |
      | Note: Run does not update operator's event
      | and can be used only with non-async
      | executors that do not rely on events
      */
    #[inline] fn run(&mut self, stream_id: i32) -> bool {
        
        todo!();
        /*
            try {
          StartAllObservers();

          context_.SwitchToDevice(stream_id);

          // Clear floating point exception flags before RunOnDevice. We will test
          // exception flags afterwards, and raise an error if an exception has
          // happened.
          if (FLAGS_caffe2_operator_throw_if_fp_exceptions ||
              FLAGS_caffe2_operator_throw_if_fp_overflow_exceptions) {
            std::feclearexcept(FE_ALL_EXCEPT);
          }

    #ifdef __GNU_LIBRARY__
          // If glibc is available, use feenableexcept that will raise exception
          // right away.
          int old_enabled_exceptions = 0;
          if (FLAGS_caffe2_operator_throw_on_first_occurrence_if_fp_exceptions) {
            if (FLAGS_caffe2_operator_throw_if_fp_exceptions ||
                FLAGS_caffe2_operator_throw_if_fp_overflow_exceptions) {
              int flag = 0;
              if (FLAGS_caffe2_operator_throw_if_fp_exceptions) {
                flag |= FE_DIVBYZERO | FE_INVALID;
              }
              if (FLAGS_caffe2_operator_throw_if_fp_overflow_exceptions) {
                flag |= FE_OVERFLOW;
              }
              old_enabled_exceptions = feenableexcept(flag);
            }
          }
    #endif
          bool result = RunOnDevice();
    #ifdef __GNU_LIBRARY__
          if (FLAGS_caffe2_operator_throw_on_first_occurrence_if_fp_exceptions) {
            if (FLAGS_caffe2_operator_throw_if_fp_exceptions ||
                FLAGS_caffe2_operator_throw_if_fp_overflow_exceptions) {
              fedisableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
              std::feclearexcept(FE_ALL_EXCEPT);
              feenableexcept(old_enabled_exceptions);
            }
          }
    #endif
          if (FLAGS_caffe2_operator_throw_if_fp_exceptions) {
            CAFFE_ENFORCE(
                !std::fetestexcept(FE_DIVBYZERO),
                "Division by zero floating point exception (FE_DIVBYZERO) reported.");
            CAFFE_ENFORCE(
                !std::fetestexcept(FE_INVALID),
                "Invalid floating point exception (FE_INVALID) reported.");
          }
          if (FLAGS_caffe2_operator_throw_if_fp_overflow_exceptions) {
            CAFFE_ENFORCE(
                !std::fetestexcept(FE_OVERFLOW),
                "Overflow floating point exception (FE_OVERFLOW) reported.");
          }
          if (!result) {
            this->RecordLastFailedOpNetPosition();
          }
          context_.FinishDeviceComputation(); // throws on error

          StopAllObservers();

          return result;
        } catch (EnforceNotMet& err) {
          if (has_debug_def()) {
            err.add_context(
                "Error from operator: \n" + ProtoDebugString(debug_def()));
            AddRelatedBlobInfo(&err);
          }
          this->RecordLastFailedOpNetPosition();
          StopAllObservers();
          throw;
        } catch (...) {
          this->RecordLastFailedOpNetPosition();
          StopAllObservers();
          throw;
        }
        */
    }
}


