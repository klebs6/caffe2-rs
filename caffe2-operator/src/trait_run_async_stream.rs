crate::ix!();

pub trait RunAsyncStream {
    
    /**
      | RunAsync, if implemented by the specific
      | operators, will schedule the computation
      | on the corresponding context and record
      | the event in its event_ member object.
      | 
      | If the specific operator does not support
      | 
      | RunAsync, it will simply be synchronous
      | as a fallback.
      |
      */
    #[inline] fn run_async_fallback(&mut self, stream_id: i32) -> bool {
        
        todo!();
        /*
            try {
        auto result = Run(stream_id);
        if (result) {
          if (HasAsyncPart()) {
            RecordEvent();
          } else {
            SetEventFinished();
          }
        } else {
          SetEventFinished(getErrorMsg().c_str());
        }
        return result;
      } catch (EnforceNotMet& err) {
        SetEventFinishedWithException(err.what());
        throw;
      } catch (const std::exception& err) {
        SetEventFinishedWithException(err.what());
        throw;
      } catch (...) {
        SetEventFinishedWithException(getErrorMsg().c_str());
        throw;
      }
        */
    }

    #[inline] fn run_async(&mut self, stream_id: i32) -> bool {
        
        todo!();
        /*
            try {
          StartAllObservers();

          context_.SwitchToDevice(stream_id);
          auto result = RunOnDevice();
          if (result) {
            if (HasAsyncPart()) {
              RecordEvent();
            } else {
              // Manually set CPU operator's event status to finished,
              // unless this is an async CPU operator
              SetEventFinished();
            }
          } else {
            SetEventFinished(getErrorMsg().c_str());
            this->RecordLastFailedOpNetPosition();
          }

          StopAllObservers();

          return result;
        } catch (EnforceNotMet& err) {
          if (has_debug_def()) {
            err.add_context(
                "Error from operator: \n" + ProtoDebugString(debug_def()));
            AddRelatedBlobInfo(&err);
          }
          SetEventFinishedWithException(err.what());
          this->RecordLastFailedOpNetPosition();
          StopAllObservers();
          throw;
        } catch (const std::exception& err) {
          SetEventFinishedWithException(err.what());
          this->RecordLastFailedOpNetPosition();
          StopAllObservers();
          throw;
        } catch (...) {
          SetEventFinishedWithException(getErrorMsg().c_str());
          this->RecordLastFailedOpNetPosition();
          StopAllObservers();
          throw;
        }
        */
    }
}
