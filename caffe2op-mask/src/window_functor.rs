crate::ix!();

pub struct WindowFunctor {
    c: *const i32,
    r: i32,
}

impl WindowFunctor {
    
    #[inline] pub fn invoke(&mut self, i: i32, j: i32, val: f32) -> bool {
        
        todo!();
        /*
            return j > c[i] + r || j < c[i] - r;
        */
    }
}
