crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/fake_quant_affine.h]

lazy_static!{
    /*
    using fake_quant_tensor_cachemask_fn = void (*)(
        Tensor& output,
        Tensor& mask,
        const Tensor& input,
        float sc,
        i64 z_point,
        i64 quant_min,
        i64 quant_max);

    using fake_quant_learnable_grad_tensor_fn = void (*)(
        TensorIterator& iter,
        float scale,
        float inv_scale,
        i64 zero_point,
        i64 quant_min,
        i64 quant_max,
        float grad_factor);
    */
}

declare_dispatch!{fake_quant_tensor_cachemask_fn, fake_quant_tensor_cachemask_stub}
declare_dispatch!{fake_quant_learnable_grad_tensor_fn, fake_quant_grad_learnable_tensor_stub}

lazy_static!{
    /*
    using fake_quant_per_channel_fn = void (*)(
        TensorIterator &iter,
        i64 quant_min,
        i64 quant_max);

    using fake_quant_per_channel_cachemask_fn = void (*)(
        TensorIterator &iter,
        TensorIterator &iter_mask,
        i64 quant_min,
        i64 quant_max);
    */
}

declare_dispatch!{fake_quant_per_channel_cachemask_fn, fake_quant_per_channel_cachemask_stub}

lazy_static!{
    /*
    using fake_quant_learnable_per_channel_fn = void (*)(
        TensorIterator &iter,
        i64 quant_min,
        i64 quant_max,
        float grad_factor);
    */
}

declare_dispatch!{fake_quant_learnable_per_channel_fn, fake_quant_grad_learnable_channel_stub}
