crate::ix!();

pub struct UpperFunctor {

}

impl UpperFunctor {
    
    #[inline] pub fn invoke(&mut self, i: i32, j: i32, val: f32) -> bool {
        
        todo!();
        /*
            return j > i;
        */
    }
}

pub struct UpperDiagFunctor {

}

impl UpperDiagFunctor {
    
    #[inline] pub fn invoke(
        &mut self, 
        i: i32, 
        j: i32, 
        val: f32) -> bool 
    {
        todo!();
        /*
            return j >= i;
        */
    }
}
