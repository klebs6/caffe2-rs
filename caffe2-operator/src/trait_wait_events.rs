crate::ix!();

pub trait WaitEvents: WaitEvent {

    //TODO is this the fallback?
    #[inline] fn wait_events_fallback(
        &mut self, 
        events:    &Vec<*const Event>,
        stream_id: Option<i32>)  
    {
        let stream_id = stream_id.unwrap_or(-1);
        
        todo!();
        /*
            for (const auto& ev : events) {
          ev->Finish();
        }
        */
    }
    
    #[inline] fn wait_events(
        &mut self, 
        events:    &Vec<*const Event>,
        stream_id: Option<i32>)  
    {
        let stream_id = stream_id.unwrap_or(-1);
        
        todo!();
        /*
            if (stream_id >= 0) {
            context_.SwitchToDevice(stream_id);
        }
        for (const auto& ev : events) {
            context_.WaitEvent(*ev);
        }
        */
    }
}
