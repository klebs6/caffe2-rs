crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/RenormKernel.cpp]

pub fn renorm_scale_factor_impl(
        iter:    &mut TensorIteratorBase,
        maxnorm: f64)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "renorm_scale_factor_cpu", [&] {
        using vec_t = at::vec::Vectorized<Scalar>;
        const auto maxnorm_s = static_cast<Scalar>(maxnorm);
        const auto maxnorm_v = vec_t(maxnorm_s);
        const auto eps_v = vec_t(static_cast<Scalar>(1e-7));
        const auto one_v = vec_t(1.0);
        cpu_kernel_vec(
          iter,
          [maxnorm_s](Scalar norm) -> Scalar {
            const auto eps = static_cast<Scalar>(1e-7);
            return (norm > maxnorm_s) ?
                maxnorm_s / (norm + eps) : static_cast<Scalar>(1.0);
          },
          [maxnorm_v, eps_v, one_v](vec_t norm) -> vec_t {
            auto fct = maxnorm_v / (norm + eps_v);
            return vec_t::blendv(one_v, fct, norm > maxnorm_v);
          });
      });
        */
}

register_dispatch!{
    renorm_scale_factor_stub, 
    &renorm_scale_factor_impl
}
