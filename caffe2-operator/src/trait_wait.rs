crate::ix!();

pub trait Wait {

    #[inline] fn wait<Context>(
        &mut self, 
        other: &OperatorStorage,
        stream_id: Option<i32>)  
    {
        let stream_id = stream_id.unwrap_or(-1);
        
        todo!();
        /*
            if (!other.IsEventDisabled()) {
          WaitEvent(other.event(), stream_id);
        }
        */
    }
}
