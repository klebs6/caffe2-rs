crate::ix!();

/**
  | ONNX wrapper functions for protobuf's
  | GetEmptyStringAlreadyInited() function used to
  | avoid duplicated global variable in the case
  | when protobuf is built with hidden visibility.
  */
#[inline] pub fn onnx_wrapper_get_empty_string_already_inited<'a>() -> &'a String {
    
    todo!();
    /*
        return ::google::protobuf::internal::GetEmptyStringAlreadyInited();
    */
}

/**
  | Caffe2 wrapper functions for protobuf's
  | GetEmptyStringAlreadyInited() function used to
  | avoid duplicated global variable in the case
  | when protobuf is built with hidden visibility.
  */
#[inline] pub fn caffe2_wrapper_get_empty_string_already_inited<'a>() -> &'a String {
    
    todo!();
    /*
        return ::google::protobuf::internal::GetEmptyStringAlreadyInited();
    */
}

/**
  | A wrapper function to shut down protobuf
  | library (this is needed in ASAN testing and
  | valgrind cases to avoid protobuf appearing to
  | "leak" memory).
  */
#[inline] pub fn shutdown_protobuf_library()  {
    
    todo!();
    /*
        ::google::protobuf::ShutdownProtobufLibrary();
    */
}
