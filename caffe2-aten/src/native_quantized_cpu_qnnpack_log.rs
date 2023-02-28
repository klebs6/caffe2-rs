crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/log.h]

#[cfg(not(PYTORCH_QNNP_LOG_LEVEL))]
lazy_static!{
    /*
    #define PYTORCH_QNNP_LOG_LEVEL CLOG_WARNING
    */
}

clog_define_log_debug!(
    pytorch_qnnp_log_debug, 
    "QNNPACK", 
    PYTORCH_QNNP_LOG_LEVEL
);

clog_define_log_info!(
    pytorch_qnnp_log_info, 
    "QNNPACK", 
    PYTORCH_QNNP_LOG_LEVEL
);

clog_define_log_warning!(
    pytorch_qnnp_log_warning, 
    "QNNPACK", 
    PYTORCH_QNNP_LOG_LEVEL
);

clog_define_log_error!(
    pytorch_qnnp_log_error, 
    "QNNPACK", 
    PYTORCH_QNNP_LOG_LEVEL
);

clog_define_log_fatal!(
    pytorch_qnnp_log_fatal, 
    "QNNPACK", 
    PYTORCH_QNNP_LOG_LEVEL
);
