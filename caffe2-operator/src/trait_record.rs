crate::ix!();

pub trait RecordEvent {

    #[inline] fn record_event_fallback(&mut self, err_msg: *const u8)  {
        
        todo!();
        /*
            CAFFE_NOT_IMPLEMENTED;
        */
    }

    #[inline] fn record_event(&mut self, err_msg: *const u8)  {
        
        todo!();
        /*
            if (event_) {
          context_.Record(event_.get(), err_msg);
        }
        */
    }
}

pub trait RecordLastFailedOpNetPosition {

    #[inline] fn record_last_failed_op_net_position(&mut self)  {
        
        todo!();
        /*
            if (net_position_ != kNoNetPositionSet) {
          VLOG(1) << "Operator with id " << net_position_ << " failed";
          operator_ws_->last_failed_op_net_position = net_position_;
        } else {
          VLOG(1) << "Failed operator doesn't have id set";
        }
        */
    }
}

