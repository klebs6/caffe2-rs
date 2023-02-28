crate::ix!();

register_context!{
    DeviceType::IDEEP, 
    caffe2::IDEEPContext
}

#[inline] pub fn copy_bytes_wrapper(
    nbytes:      usize,
    src:         *const c_void,
    src_device:  Device,
    dst:         *mut c_void,
    dst_device:  Device)  
{
    todo!();
    /*
        if (nbytes == 0) {
        return;
      }
      CAFFE_ENFORCE(src);
      CAFFE_ENFORCE(dst);
      memcpy(dst, src, nbytes);
    */
}

register_copy_bytes_function!{
    DeviceType::IDEEP,
    DeviceType::CPU,
    CopyBytesWrapper
}

register_copy_bytes_function!{
    DeviceType::CPU,
    DeviceType::IDEEP,
    CopyBytesWrapper
}

register_copy_bytes_function!{
    DeviceType::IDEEP,
    DeviceType::IDEEP,
    CopyBytesWrapper
}

caffe_known_type!{ideep::tensor}

define_registry!{
    IDEEPOperatorRegistry,
    OperatorStorage,
    OperatorDef,
    Workspace
}

caffe_register_device_type!{
    DeviceType::IDEEP, 
    IDEEPOperatorRegistry 
}

register_event_create_function!{IDEEP,         EventCreateCPU}
register_event_record_function!{IDEEP,         EventRecordCPU}
register_event_wait_function!{IDEEP,           IDEEP, EventWaitCPUCPU}
register_event_wait_function!{IDEEP,           CPU,   EventWaitCPUCPU}
register_event_wait_function!{CPU,             IDEEP, EventWaitCPUCPU}
register_event_finish_function!{IDEEP,         EventFinishCPU}
register_event_query_function!{IDEEP,          EventQueryCPU}
register_event_error_message_function!{IDEEP,  EventErrorMessageCPU}
register_event_set_finished_function!{IDEEP,   EventSetFinishedCPU}
register_event_reset_function!{IDEEP,          EventResetCPU}
