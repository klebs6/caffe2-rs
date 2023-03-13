crate::ix!();

pub struct Params {

    /**
      | return all key-frames regardless of
      | specified fps
      |
      */
    key_frames:                bool, // default = false

    /// return audio data while decoding the video
    get_audio:                 bool, // default = false

    /// for sampling audio data
    outrate:                   i32, // default = 22000
    outfmt:                    i32, // default = AV_SAMPLE_FMT_FLT
    outlayout:                 i64, // default = AV_CH_LAYOUT_MONO

    // Output image pixel format
    pixel_format:              AVPixelFormat, // default = AVPixelFormat::AV_PIX_FMT_RGB24

    /**
      | Index of stream to decode.
      |
      | -1 will automatically decode the first
      | video stream.
      */
    stream_index:              i32, // default = -1

    /**
      | How many frames to output at most from
      | the video -1 no limit
      |
      */
    maximum_output_frames:     i32, // default = -1

    /// params for video resolution
    video_res_type:            i32, // default = VideoResType::USE_WIDTH_HEIGHT
    crop_size:                 i32, // default = -1
    short_edge:                i32, // default = -1

    /**
      | Output video size, -1 to preserve origianl
      | dimension
      |
      */
    output_width:              i32, // default = -1
    output_height:             i32, // default = -1

    /**
      | max output dimension, -1 to preserve
      | original size
      |
      | the larger dimension of the video will be
      | scaled to this size, and the second
      | dimension will be scaled to preserve
      | aspect ratio
      */
    max_output_dimension:      i32, // default = -1

    /// params for decoding behavior
    decode_type:               i32, // default = DecodeType::DO_TMP_JITTER
    num_of_required_frame:     i32, // default = -1

    /**
      | intervals_ control variable sampling fps
      | between different timestamps
      |
      | intervals_ must be ordered strictly
      | ascending by timestamps
      |
      | the first interval must have a timestamp
      | of zero
      |
      | fps must be either the 3 special fps
      | defined in SpecialFps, or > 0
      | = {{0, SpecialFps::SAMPLE_ALL_FRAMES}};
      */
    intervals:                 Vec<SampleInterval>,
}

impl Params {
    
    /**
      | FPS of output frames setting here will
      | reset intervals_ and force decoding
      | at target FPS
      | 
      | This can be used if user just want to decode
      | at a steady fps
      |
      */
    #[inline] pub fn fps(&mut self, v: f32) -> &mut Params {
        
        todo!();
        /*
            intervals_.clear();
        intervals_.emplace_back(0, v);
        return *this;
        */
    }
    
    /**
      | Sample output frames at a specified
      | list of timestamps
      | 
      | Timestamps must be in increasing order,
      | and timestamps past the end of the video
      | will be ignored
      | 
      | Setting here will reset intervals_
      |
      */
    #[inline] pub fn set_sample_timestamps(&mut self, timestamps: &Vec<f64>) -> &mut Params {
        
        todo!();
        /*
            intervals_.clear();
        // insert an interval per desired frame.
        for (auto& timestamp : timestamps) {
          intervals_.emplace_back(timestamp, SpecialFps::SAMPLE_TIMESTAMP_ONLY);
        }
        return *this;
        */
    }
    
    /**
      | Pixel format of output buffer, default
      | PIX_FMT_RGB24
      |
      */
    #[inline] pub fn pixel_format(&mut self, pixel_format: AVPixelFormat) -> &mut Params {
        
        todo!();
        /*
            pixelFormat_ = pixelFormat;
        return *this;
        */
    }
    
    /**
      | Return all key-frames
      |
      */
    #[inline] pub fn key_frames(&mut self, key_frames: bool) -> &mut Params {
        
        todo!();
        /*
            keyFrames_ = keyFrames;
        return *this;
        */
    }
    
    /**
      | Index of video stream to process, defaults
      | to the first video stream
      |
      */
    #[inline] pub fn stream_index(&mut self, index: i32) -> &mut Params {
        
        todo!();
        /*
            streamIndex_ = index;
        return *this;
        */
    }

    /**
      | Only output this many frames, default
      | to no limit
      |
      */
    #[inline] pub fn max_output_frames(&mut self, count: i32) -> &mut Params {
        
        todo!();
        /*
            maximumOutputFrames_ = count;
        return *this;
        */
    }
    
    /**
      | Output frame width, default to video
      | width
      |
      */
    #[inline] pub fn output_width(&mut self, width: i32) -> &mut Params {
        
        todo!();
        /*
            outputWidth_ = width;
        return *this;
        */
    }

    /**
      | Output frame height, default to video
      | height
      |
      */
    #[inline] pub fn output_height(&mut self, height: i32) -> &mut Params {
        
        todo!();
        /*
            outputHeight_ = height;
        return *this;
        */
    }
    
    /**
      | Max dimension of either width or height,
      | if any is bigger it will be scaled down
      | to this and econd dimension will be scaled
      | down to maintain aspect ratio.
      |
      */
    #[inline] pub fn max_output_dimension(&mut self, size: i32) -> &mut Params {
        
        todo!();
        /*
            maxOutputDimension_ = size;
        return *this;
        */
    }
}
