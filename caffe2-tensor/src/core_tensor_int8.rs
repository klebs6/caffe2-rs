crate::ix!();


pub struct Int8TensorCPU {
    scale:      f32, // default = 1.0
    zero_point: i32, // default = 0

    /// Generally stores uint8_t data, but
    /// sometimes int32_t (e.g. bias parameters).
    t: Tensor, // default = CPU
}

caffe_known_type!{Int8TensorCPU}
