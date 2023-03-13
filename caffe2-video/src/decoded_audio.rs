crate::ix!();

/// data structure for storing decoded audio data
///
pub struct DecodedAudio<'a> {
    data_size:       i32,
    out_sample_size: i32,
    audio_data:      Box<&'a [f32]>,
}

impl<'a> DecodedAudio<'a> {
    
    pub fn new(
        data_size:       i32,
        out_sample_size: i32,
        audio_data:      Box<&[f32]>) -> Self {
        todo!();
        /*
            : dataSize_(dataSize),
            outSampleSize_(outSampleSize),
            audio_data_(std::move(audio_data))
        */
    }
}
