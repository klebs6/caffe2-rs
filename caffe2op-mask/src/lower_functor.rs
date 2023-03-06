crate::ix!();

pub struct LowerFunctor {

}

impl LowerFunctor {
    
    #[inline] pub fn invoke(&mut self, i: i32, j: i32, val: f32) -> bool {
        
        todo!();
        /*
            return j < i;
        */
    }
}

pub struct LowerDiagFunctor;

impl LowerDiagFunctor {

    #[inline] pub fn invoke(&mut self, i: i32, j: i32, val: f32) -> bool {
        
        todo!();
        /*
            return j <= i;
        */
    }
}
