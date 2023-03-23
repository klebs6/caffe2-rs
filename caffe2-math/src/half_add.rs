crate::ix!();

pub struct HalfAddFunctor { }

impl HalfAddFunctor {
    
    #[inline] pub fn invoke(&self, a: f16, b: f16) -> f16 {
        
        todo!();
        /*
            return convert::To<f32, Half>(
            convert::To<Half, f32>(a) + convert::To<Half, f32>(b));
        */
    }
}
