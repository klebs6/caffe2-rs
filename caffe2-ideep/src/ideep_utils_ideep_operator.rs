crate::ix!();

declare_registry!{
    IDEEPOperatorRegistry,
    OperatorStorage,
    OperatorDef,
    Workspace
}

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

/**
  | Convert zero_point scales to min_max scales
  | NOTE:
  |
  |  The scales in operator is saved in FBGEMM
  |  format, while FBGEMM scales are the
  |  reciprocals of MKL-DNN scales.
  |
  |  This function is provided to convert scales
  |  from FBGEMM to MKL-DNN
  */
#[inline] pub fn convert_scales(scales_z: Vec<f32>) -> IDEEPScale {
    
    todo!();
    /*
        ideep::scale_t scales (scales_z);
      for (auto it = scales.begin(); it != scales.end(); it++) {
        *it = 1.0f / *it;
      }
      return scales;
    */
}

#[inline] pub fn canonical_dims(
    adims: IDEEPTensorDims,
    axis: i32) -> IDEEPTensorDims 
{
    todo!();
    /*
        CAFFE_ENFORCE(axis < (int32_t)adims.size(), "Invalid axis!");
      CAFFE_ENFORCE(axis > (int32_t)-adims.size(), "Invalid axis!");
      if (adims.size() == 2 || axis == 1)
        return adims;
      if (axis < 0) {
        axis += (int32_t)adims.size();
      }

      auto dim0 = std::accumulate(adims.begin(), adims.begin() + axis, 1,
                                  std::multiplies<ideep::tensor::dim_t>());
      auto dim1 = std::accumulate(adims.begin() + axis, adims.end(), 1,
                                  std::multiplies<ideep::tensor::dim_t>());
      return ideep::tensor::dims({dim0, dim1});
    */
}
