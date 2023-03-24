crate::ix!();

pub trait SetEventFinished {

    #[inline] fn set_event_finished(&mut self, err_msg: *const u8)  {
        
        todo!();
        /*
            if (event_) {
          event_->SetFinished(err_msg);
        }
        */
    }

    #[inline] fn set_event_finished_with_exception(&mut self, err_msg: *const u8)  {
        
        todo!();
        /*
            if (event_) {
          event_->SetFinishedWithException(err_msg);
        }
        */
    }
}
