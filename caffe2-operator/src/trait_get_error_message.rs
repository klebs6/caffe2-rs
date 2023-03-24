crate::ix!();

pub trait GetErrorMessage {

    #[inline] fn get_error_msg(&mut self) -> String {
        
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
