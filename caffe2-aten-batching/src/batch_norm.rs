crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/batch_norm.h]

lazy_static!{
    /*
    using batch_norm_fn = void (*)(Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, bool, double);
    using batch_norm_collect_stats_fn = void (*)(Tensor&, Tensor&, const Tensor&);
    using batch_norm_backward_fn = void(*)(Tensor&, Tensor&, Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, bool, double);
    */
}

declare_dispatch!{batch_norm_fn               , batch_norm_cpu_stub}
declare_dispatch!{batch_norm_collect_stats_fn , batch_norm_cpu_collect_stats_stub}
declare_dispatch!{batch_norm_backward_fn      , batch_norm_cpu_backward_stub}

/// TensorAccessor when it is defined to work
/// around undefined...
///
pub fn conditional_accessor_1d<Scalar>(t: &Tensor) -> TensorAccessor<Scalar,1> {

    todo!();
        /*
            if (! t.defined()) {
        return TensorAccessor<Scalar, 1>(nullptr, nullptr, nullptr);
      }
      return t.accessor<Scalar, 1>();
        */
}

pub fn conditional_data_ptr<Scalar>(t: &Tensor) -> *mut Scalar {

    todo!();
        /*
            return t.defined() ? t.contiguous().data_ptr<Scalar>()
                         : nullptr;
        */
}
