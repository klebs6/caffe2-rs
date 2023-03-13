crate::ix!();

pub struct VideoMeta {
    fps:         f64,
    width:       i32,
    height:      i32,
    codec_type:  AVMediaType,
    pix_format:  AVPixelFormat,
}

impl Default for VideoMeta {
    
    fn default() -> Self {
        todo!();
        /*
            : fps(-1),
            width(-1),
            height(-1),
            codec_type(AVMEDIA_TYPE_VIDEO),
            pixFormat(AVPixelFormat::AV_PIX_FMT_RGB24
        */
    }
}
