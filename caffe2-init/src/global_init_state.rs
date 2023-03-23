crate::ix!();

/**
  | Keep track of stages of initialization
  | to differentiate between
  | 
  | (a) Re-entrant calls to GlobalInit
  | (e.g. caller registers a Caffe2 init
  | function which might in turn indirectly
  | invoke GlobalInit).
  | 
  | (b) Successive calls to GlobalInit,
  | which are handled as documented in init.h.
  | 
  | -----------
  | @note
  | 
  | this is NOT attempting to address thread-safety,
  | see comments in init.h.
  |
  */
pub enum State {
    Uninitialized,
    Initializing,
    Initialized,
}

#[inline] pub fn global_init_state<'a>() -> &'a mut State {
    
    todo!();
    /*
        static State state = State::Uninitialized;
                return state;
    */
}


