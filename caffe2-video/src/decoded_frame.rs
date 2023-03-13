crate::ix!();

/// data structure for storing decoded video frames
///
pub struct DecodedFrame {

    /// decoded data buffer
    data:                 AvDataPtr,

    /// size in bytes
    size:                 i32, // default = 0

    /// frame dimensions
    width:                i32, // default = 0
    height:               i32, // default = 0

    /**
      | timestamp in seconds since beginning
      | of video
      |
      */
    timestamp:            f64, // default = 0

    /// true if this is a key frame.
    key_frame:            bool, // default = false

    /// index of frame in video
    index:                i32, // default = -1

    /// Sequential number of outputted frame
    output_frame_index:   i32, // default = -1
}

