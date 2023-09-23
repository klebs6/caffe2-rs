crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ReduceAllOps.h]

lazy_static!{
    /*
    using reduce_all_fn = void (*)(Tensor & result, const Tensor & self);
    using reduce_min_max_fn = void (*)(Tensor & max_result, Tensor & min_result, const Tensor & self);
    */
}

declare_dispatch!{reduce_all_fn, min_all_stub}
declare_dispatch!{reduce_all_fn, max_all_stub}
declare_dispatch!{reduce_min_max_fn, _aminmax_all_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ReduceAllOps.cpp]

define_dispatch!{min_all_stub}
define_dispatch!{max_all_stub}
define_dispatch!{_aminmax_all_stub}

pub fn min(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.numel() > 0,
                  "min(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");
      Tensor result = empty({}, self.options());
      min_all_stub(self.device().type(), result, self.contiguous());
      return result;
        */
}

pub fn max(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.numel() > 0,
                  "max(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");
      Tensor result = empty({}, self.options());
      max_all_stub(self.device().type(), result, self.contiguous());
      return result;
        */
}

pub fn aminmax_all(self_: &Tensor) -> (Tensor,Tensor) {
    
    todo!();
        /*
            TORCH_CHECK(self.numel() > 0,
                  "_aminmax_all(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");
      Tensor min_result = empty({}, self.options());
      Tensor max_result = empty({}, self.options());
      _aminmax_all_stub(self.device().type(), min_result, max_result, self.contiguous());
      return tuple<Tensor&, Tensor&>(min_result, max_result);
        */
}
