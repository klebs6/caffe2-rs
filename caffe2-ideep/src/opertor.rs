crate::ix!();

/**
  | IDEEPOperator is the base scaffolding
  | of the operators that uses IDEEP. It
  | provides a few operators that are useful
  | to IDEEP specific implementations.
  |
  */
pub struct IDEEPOperator {
    base:    OperatorStorage,
    context: IDEEPContext,
    order:   StorageOrder,
}  

impl IDEEPOperator {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : OperatorStorage(operator_def, ws),
            context_(operator_def.device_option()),
            order_(StringToStorageOrder(
                OperatorStorage::GetSingleArgument<string>("order", "NCHW")))
        */
    }
    
    #[inline] pub fn input(&mut self, index: i32) -> &IDEEPTensor {
        
        todo!();
        /*
            return OperatorStorage::template Input<ideep::tensor>(index);
        */
    }
    
    #[inline] pub fn output(&mut self, index: i32) -> *mut IDEEPTensor {
        
        todo!();
        /*
            return OperatorStorage::template Output<ideep::tensor>(index);
        */
    }
    
    /**
      | The run function of Operator switches to
      | the device, and then carries out the
      | actual computation with RunOnDevice(). You
      | should implement RunOnDevice instead of
      | Run().
      */
    #[inline] pub fn run(&mut self, stream_id: i32) -> bool {
        
        todo!();
        /*
            // Since IDEEP does not need to do SwithToDevice and
        // FinishDeviceComputation,
        // it is always just a re-route to RunOnDevice().
        try {
          StartAllObservers();
          bool result =  RunOnDevice();
          StopAllObservers();
          return result;
        } catch (EnforceNotMet& err) {
          TORCH_RETHROW(err, getErrorMsg());
        } catch (ideep::error& e) {
          LOG(ERROR) << "IDEEP error:" << e.message;
          throw;
        }
        */
    }

    /**
      | Waits for a previous event. Note that to
      | properly wait and run asynchronously,
      | WaitEvent, RunAsync and Record should all
      | be executed on the same CPU thread.
      */
    #[inline] pub fn wait_event(&mut self, ev: &Event, unused: i32)  {
        
        todo!();
        /*
            context_.WaitEvent(ev);
        */
    }
    
    #[inline] pub fn wait_events(&mut self, events: &Vec<*const Event>, unused: i32)  {
        
        todo!();
        /*
            for (const auto& ev : events) {
          context_.WaitEvent(*ev);
        }
        */
    }
    
    #[inline] pub fn record_event(&mut self, err_msg: *const u8)  {

        todo!();
        /*
            if (event_) {
          context_.Record(event_.get(), err_msg);
        }
        */
    }
    
    #[inline] pub fn get_error_msg(&mut self) -> String {
        
        todo!();
        /*
            if (has_debug_def()) {
          return "Error from operator: " + ProtoDebugString(debug_def());
        } else {
          return "Error from operator: no op def";
        }
        */
    }
}
