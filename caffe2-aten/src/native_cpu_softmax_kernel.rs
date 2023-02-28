crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/SoftmaxKernel.h]

lazy_static!{
    /*
    using forward_fn = void(*)(Tensor &, const Tensor &);
    using backward_fn = void(*)(Tensor &, const Tensor &, const Tensor&);
    using forward_fn_with_dim = void(*)(Tensor &, const Tensor &, const i64);
    */
}

declare_dispatch!{forward_fn, softmax_lastdim_kernel}
declare_dispatch!{forward_fn, log_softmax_lastdim_kernel}
declare_dispatch!{backward_fn, softmax_backward_lastdim_kernel}
declare_dispatch!{backward_fn, log_softmax_backward_lastdim_kernel}
declare_dispatch!{forward_fn_with_dim, softmax_kernel}
