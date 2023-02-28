/*!
  | Using google glog. For glog 0.3.2 versions,
  | stl_logging.h needs to be before logging.h to
  | actually use stl_logging. Because template
  | magic.
  |
  | In addition, we do not do stl logging in .cu
  | files because nvcc does not like it.
  |
  | Some mobile platforms do not like stl_logging,
  | so we add an overload in that case as well.
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/logging_is_google_glog.h]

/**
  | In the cudacc compiler scenario, we will simply
  | ignore the container printout feature.
  |
  | Basically we need to register a fake overload
  | for vector/string - here, we just ignore the
  | entries in the logs.
  */
#[cfg(not(all(not(__CUDACC__),not(C10_USE_MINIMAL_GLOG))))]
macro_rules! INSTANTIATE_FOR_CONTAINER {
    ($container:ident) => {
        /*
        
          template <class... Types>                                       
          ostream& operator<<(ostream& out, const container<Types...>&) { 
            return out;                                                   
          }
        */
    }
}

#[cfg(not(all(not(__CUDACC__),not(C10_USE_MINIMAL_GLOG))))] INSTANTIATE_FOR_CONTAINER!{vector}
#[cfg(not(all(not(__CUDACC__),not(C10_USE_MINIMAL_GLOG))))] INSTANTIATE_FOR_CONTAINER!{map}
#[cfg(not(all(not(__CUDACC__),not(C10_USE_MINIMAL_GLOG))))] INSTANTIATE_FOR_CONTAINER!{set}

// Additional macros on top of glog

/**
  | Log with source location information override
  | (to be used in generic warning/error handlers
  | implemented as functions, not macros)
  |
  | Note, we don't respect GOOGLE_STRIP_LOG here
  | for simplicity
  */
macro_rules! LOG_AT_FILE_LINE {
    ($n:ident, $file:ident, $line:ident) => {
        /*
        
          ::google::LogMessage(file, line, ::google::GLOG_##n).stream()
        */
    }
}
