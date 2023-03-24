crate::ix!();

pub trait CheckStream {

    #[inline] fn is_stream_free(&self, stream_id: i32) -> bool {
        
        todo!();
        /*
            return context_.IsStreamFree(device_option(), stream_id);
        */
    }
}
