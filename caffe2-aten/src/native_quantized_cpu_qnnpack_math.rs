crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/math.h]

#[inline] pub fn min(a: Size, b: Size) -> Size {
    
    todo!();
        /*
            return a < b ? a : b;
        */
}

#[inline] pub fn max(a: Size, b: Size) -> Size {
    
    todo!();
        /*
            return a > b ? a : b;
        */
}

#[inline] pub fn doz(a: Size, b: Size) -> Size {
    
    todo!();
        /*
            return a < b ? 0 : a - b;
        */
}

#[inline] pub fn divide_round_up(n: Size, q: Size) -> Size {
    
    todo!();
        /*
            return n % q == 0 ? n / q : n / q + 1;
        */
}

#[inline] pub fn round_up(n: Size, q: Size) -> Size {
    
    todo!();
        /*
            return divide_round_up(n, q) * q;
        */
}
