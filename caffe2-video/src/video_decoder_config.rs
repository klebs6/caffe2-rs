crate::ix!();

pub const VIO_BUFFER_SZ:       usize = 32768;
pub const MAX_DECODING_FRAMES: usize = 10000;

/**
  | enum to specify 3 special fps sampling
  | behaviors: 0: disable fps sampling,
  | no frame sampled at all
  | 
  | -1: unlimited fps sampling, will sample
  | at native video fps
  | 
  | -2: disable fps sampling, but will get
  | the frame at specific timestamp
  |
  */
pub enum SpecialFps {
    SAMPLE_NO_FRAME       = 0,
    SAMPLE_ALL_FRAMES     = -1,
    SAMPLE_TIMESTAMP_ONLY = -2,
}

/**
  | three different types of resolution
  | when decoding the video
  | 
  | 0: resize to width x height and ignore
  | the aspect ratio;
  | 
  | 1: resize to short_edge and keep the
  | aspect ratio;
  | 
  | 2: using the original resolution of
  | the video; if resolution is smaller
  | than crop_size x crop_size, resize
  | to crop_size and keep the aspect ratio;
  | 
  | 3: for xray video service
  |
  */
pub enum VideoResType {
    USE_WIDTH_HEIGHT = 0,
    USE_SHORT_EDGE   = 1,
    ORIGINAL_RES     = 2,
}

/**
  | three different types of decoding behavior
  | are supported
  | 
  | 0: do temporal jittering to sample
  | a random clip from the video
  | 
  | 1: uniformly sample multiple clips
  | from the video;
  | 
  | 2: sample a clip from a given starting
  | frame
  | 
  | 3: for xray video service
  |
  */
pub enum DecodeType {
    DO_TMP_JITTER  = 0,
    DO_UNIFORM_SMP = 1,
    USE_START_FRM  = 2,
}

pub struct AvData {
    data: *mut u8,
}

impl Drop for AvData {
    fn drop(&mut self) {
        //ffmpeg_sys::av_free(self.data);
    }
}

pub type AvDataPtr = Box<AvData>;

