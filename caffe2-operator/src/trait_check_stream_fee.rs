crate::ix!();

pub trait CheckStreamFree {

    /**
      | Checks whether stream is ready to execute
      | new computation, used in stream allocation
      | optimization to skip stream that is
      | currently busy. Depends on context and
      | operator's device, returns true by default
      */
    #[inline] fn is_stream_free(&self, unused: i32) -> bool {
        
        todo!();
        /*
            return true;
        */
    }
}
