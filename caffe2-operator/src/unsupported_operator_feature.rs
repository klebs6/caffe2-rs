crate::ix!();

/**
  | An exception that can be thrown by an
  | operator constructor that notifies
  | that it does not support the given setting.
  | This can be usually used for specific
  | engines that only implement a subset
  | of the features required by the original
  | operator schema.
  | 
  | TODO(jiayq): make more feature-complete
  | exception message.
  |
  */
pub struct UnsupportedOperatorFeature {
    msg: String,
}

/**
  | A helper macro that should ONLY be used
  | in the operator constructor to check
  | if needed features are met. If not, throws
  | the UnsupportedOperatorFeature exception
  | with the given message.
  |
  */
#[macro_export] macro_rules! operator_needs_feature {
    ($condition:ident, $($arg:ident),*) => {
        /*
        
          if (!(condition)) {                                          
            throw UnsupportedOperatorFeature(::c10::str(__VA_ARGS__)); 
          }
        */
    }
}
