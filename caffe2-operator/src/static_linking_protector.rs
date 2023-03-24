crate::ix!();

/**
  | StaticLinkingProtector is a helper
  | class that ensures that the Caffe2 library
  | is linked correctly with whole archives
  | (in the case of static linking). What
  | happens is that when
  | 
  | CreateOperator is called for the first
  | time, it instantiates an OperatorLinkingProtector
  | object to check if the operator registry
  | is empty. If it is empty, this means that
  | we are not properly linking the library.
  | 
  | You should not need to use this class.
  |
  */
pub struct StaticLinkingProtector {
    
}

impl Default for StaticLinkingProtector {
    
    fn default() -> Self {
        todo!();
        /*
            const auto registered_ops = CPUOperatorRegistry()->Keys().size();
        // Note: this is a check failure instead of an exception, because if
        // the linking is wrong, Caffe2 won't be able to run properly anyway,
        // so it's better to fail loud.
        // If Caffe2 is properly linked with whole archive, there should be more
        // than zero registered ops.
        if (registered_ops == 0) {
          LOG(FATAL)
              << "You might have made a build error: the Caffe2 library does not seem "
                 "to be linked with whole-static library option. To do so, use "
                 "-Wl,-force_load (clang) or -Wl,--whole-archive (gcc) to link the "
                 "Caffe2 library.";
        
        */
    }
}
