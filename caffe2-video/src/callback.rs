crate::ix!();

pub trait Callback { 
    fn frame_decoded(&mut self,          img: Box<DecodedFrame>);
    fn audio_decoded(&mut self,          decoded_audio_data: Box<DecodedAudio>) {}
    fn video_decoding_started(&mut self, video_meta: &VideoMeta)       {}
    fn video_decoding_ended(&mut self,   last_frame_timestamp: f64)      {}
}

pub struct CallbackImpl<'a> {
    frames:        Vec<Box<DecodedFrame>>,
    audio_samples: Vec<Box<DecodedAudio<'a>>>,
}

impl<'a> CallbackImpl<'a> {
    
    pub fn new() -> Self {
        todo!();
        /*
            clear();
        */
    }
    
    #[inline] pub fn clear(&mut self)  {
        
        todo!();
        /*
            FreeDecodedData(frames, audio_samples);
        */
    }
}

impl<'a> Callback for CallbackImpl<'a> {
    
    #[inline] fn frame_decoded(&mut self, frame: Box<DecodedFrame>)  {
        
        todo!();
        /*
            frames.push_back(move(frame));
        */
    }
    
    #[inline] fn audio_decoded(&mut self, audio_sample: Box<DecodedAudio>)  {
        
        todo!();
        /*
            audio_samples.push_back(move(audio_sample));
        */
    }
    
    #[inline] fn video_decoding_started(&mut self, video_meta: &VideoMeta)  {
        
        todo!();
        /*
            clear();
        */
    }
}
