crate::ix!();

pub trait GetEvent {

    #[inline] fn event(&self) -> &Event {
        
        todo!();
        /*
            CAFFE_ENFORCE(event_, "Event is disabled");
        return *event_;
        */
    }
}

pub trait GetEventMut {

    #[inline] fn event_mut<'a>(&'a mut self) -> &'a mut Event {
        
        todo!();
        /*
            CAFFE_ENFORCE(event_, "Event is disabled");
        return *event_;
        */
    }
}

pub trait ResetEvent {
    #[inline] fn reset_event(&mut self)  {
        
        todo!();
        /*
            if (event_) {
          event_->Reset();
        }
        */
    }
}

pub trait DisableEvent {

    #[inline] fn disable_event(&mut self)  {
        
        todo!();
        /*
            event_ = nullptr;
        */
    }
}

pub trait CheckEventDisabled {

    #[inline] fn is_event_disabled(&self) -> bool {
        
        todo!();
        /*
            return !event_;
        */
    }
}
