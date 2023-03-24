crate::ix!();

pub trait WaitEvent {

    //TODO is this the fallback?
    #[inline] fn wait_event_fallback(&mut self, ev: &Event, stream_id: Option<i32>)  {
        let stream_id = stream_id.unwrap_or(-1);
        
        todo!();
        /*
            ev.Finish();
        */
    }

    #[inline] fn wait_event(
        &mut self, 
        ev:        &Event,
        stream_id: i32)  
    {
        todo!();
        /*
            if (stream_id >= 0) {
          context_.SwitchToDevice(stream_id);
        }
        context_.WaitEvent(ev);
        */
    }
}
