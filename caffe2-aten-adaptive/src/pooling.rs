crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/AdaptivePooling.h]

lazy_static!{
    /*
    using adaptive_avg_pooling_fn = void(*)(Tensor& output, const Tensor& input, IntArrayRef output_size);
    using adaptive_avg_pooling_backward_fn = void(*)(Tensor& grad_input, const Tensor& grad_output);
    */
}

declare_dispatch!{adaptive_avg_pooling_fn, adaptive_avg_pool2d_kernel}
declare_dispatch!{adaptive_avg_pooling_backward_fn, adaptive_avg_pool2d_backward_kernel}

#[inline] pub fn start_index(
        a: i64,
        b: i64,
        c: i64) -> i64 {
    
    todo!();
        /*
            return (i64)floor((float)(a * c) / b);
        */
}

#[inline] pub fn end_index(
        a: i64,
        b: i64,
        c: i64) -> i64 {
    
    todo!();
        /*
            return (i64)ceil((float)((a + 1) * c) / b);
        */
}
