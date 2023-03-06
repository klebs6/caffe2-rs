crate::ix!();

pub struct SequenceFunctor {
    sl: *const i32,
    len: usize,
}

impl SequenceFunctor {
    
    #[inline] pub fn invoke(&mut self, i: i32, j: i32, val: f32) -> bool {
        
        todo!();
        /*
            CAFFE_ENFORCE(i < len_, "Out of bound.");
        return j >= sl_[i];
        */
    }
}
