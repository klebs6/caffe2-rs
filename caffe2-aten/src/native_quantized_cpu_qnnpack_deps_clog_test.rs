crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/deps/clog/test/clog.cc]

clog_define_log_debug!{named_log_debug          , "Unit Test" , CLOG_DEBUG}
clog_define_log_info!{named_log_info            , "Unit Test" , CLOG_INFO}
clog_define_log_warning!{named_log_warning      , "Unit Test" , CLOG_WARNING}
clog_define_log_error!{named_log_error          , "Unit Test" , CLOG_ERROR}
clog_define_log_fatal!{named_log_fatal          , "Unit Test" , CLOG_FATAL}
clog_define_log_debug!{nameless_log_debug       , NULL        , CLOG_DEBUG}
clog_define_log_info!{nameless_log_info         , NULL        , CLOG_INFO}
clog_define_log_warning!{nameless_log_warning   , NULL        , CLOG_WARNING}
clog_define_log_error!{nameless_log_error       , NULL        , CLOG_ERROR}
clog_define_log_fatal!{nameless_log_fatal       , NULL        , CLOG_FATAL}
clog_define_log_debug!{suppressed_log_debug     , NULL        , CLOG_INFO}
clog_define_log_info!{suppressed_log_info       , NULL        , CLOG_WARNING}
clog_define_log_warning!{suppressed_log_warning , NULL        , CLOG_ERROR}
clog_define_log_error!{suppressed_log_error     , NULL        , CLOG_FATAL}
clog_define_log_fatal!{suppressed_log_fatal     , NULL        , CLOG_NONE}

#[test] fn CLOG_debug() {
    todo!();
    /*
    
          named_log_debug("test debug message with a module name");
          nameless_log_debug("test debug message without a module name");
          suppressed_log_debug("test suppressed debug message");
        
    */
}

#[test] fn CLOG_info() {
    todo!();
    /*
    
          named_log_info("test info message with a module name");
          nameless_log_info("test info message without a module name");
          suppressed_log_info("test suppressed info message");
        
    */
}

#[test] fn CLOG_warning() {
    todo!();
    /*
    
          named_log_warning("test warning message with a module name");
          nameless_log_warning("test warning message without a module name");
          suppressed_log_warning("test suppressed warning message");
        
    */
}

#[test] fn CLOG_error() {
    todo!();
    /*
    
          named_log_error("test error message with a module name");
          nameless_log_error("test error message without a module name");
          suppressed_log_error("test suppressed error message");
        
    */
}
