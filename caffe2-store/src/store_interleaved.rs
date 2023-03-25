crate::ix!();

#[cfg(target_feature = "neon")]
pub fn store_interleaved_arm<const N: i32>(p: *mut f32, v: &[float32x4]) {
    /*
    match N {
        1 => store_interleaved_arm1(p, v),
        2 => store_interleaved_arm2(p, v),
        3 => store_interleaved_arm3(p, v),
        4 => store_interleaved_arm4(p, v),
        _ => unimplemented!(),
    }
    */
}

pub fn store_interleaved<const N: i32>(p: *mut f32, v: &[f32]) {
    /*
    match N {
        1 => store_interleaved1(p, v),
        2 => store_interleaved2(p, v),
        3 => store_interleaved3(p, v),
        4 => store_interleaved4(p, v),
        _ => unimplemented!(),
    }
    */
}

pub fn store_interleaved1(p: *mut f32, v: &[f32]) {
    /*
    p[0] = v[0];
    */
}

#[cfg(target_feature = "neon")]
pub fn store_interleaved_arm1(p: *mut f32, v: &[float32x4]) {
    /*
    vst1q_f32(p, v[0]);
    */
}

pub fn store_interleaved2(p: *mut f32, v: &[f32]) {
    /*
    p[0] = v[0];
    p[1] = v[1];
    */
}

#[cfg(target_feature = "neon")]
pub fn store_interleaved_arm2(p: *mut f32, v: &[float32x4]) {
    /*
    float32x4x2_t x = {{v[0], v[1]}};
    vst2q_f32(p, x);
    */
}

pub fn store_interleaved3(p: *mut f32, v: &[f32]) {
    /*
    p[0] = v[0];
    p[1] = v[1];
    p[2] = v[2];
    */
}

#[cfg(target_feature = "neon")]
pub fn store_interleaved_arm3(p: *mut f32, v: &[float32x4]) {
    /*
    float32x4x3_t x = {{v[0], v[1], v[2]}};
    vst3q_f32(p, x);
    */
}

pub fn store_interleaved4(p: *mut f32, v: &[f32]) {
    /*
    p[0] = v[0];
    p[1] = v[1];
    p[2] = v[2];
    p[3] = v[3];
    */
}

#[cfg(target_feature = "neon")]
pub fn store_interleaved_arm4(p: *mut f32, v: &[float32x4]) {
    /*
    float32x4x4_t x = {{v[0], v[1], v[2], v[3]}};
    vst4q_f32(p, x);
    */
}
