crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/CatKernel.h]

lazy_static!{
    /*
    using cat_serial_fn = void(*)(Tensor &, TensorList, i64);
    */
}

declare_dispatch!{cat_serial_fn, cat_serial_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/CatKernel.cpp]

pub struct InputMeta {
    data_ptr:   *mut void,
    inner_size: i64,
}

impl InputMeta {
    
    pub fn new(
        t:     &Tensor,
        dim:   i64,
        inner: i64) -> Self {
    
        todo!();
        /*


            : data_ptr(t.data_ptr())
        , inner_size(t.sizes()[dim] * inner)
        */
    }
}

pub fn cat_serial_kernel_impl<Scalar>(
        result:  &mut Tensor,
        tensors: TensorList,
        dim:     i64)  {

    todo!();
        /*
            TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          dim >= 0 && dim < result.dim(), "dim out of range in cat_serial_kernel_impl");
      i64 outer = result.numel() / (result.sizes()[dim] * result.strides()[dim]);
      Scalar* result_data = result.data_ptr<Scalar>();
      i64 ninputs = tensors.size();
      vector<InputMeta> inputs;
      inputs.reserve(ninputs);
      for (auto const &tensor : tensors) {
        inputs.emplace_back(tensor, dim, result.strides()[dim]);
      }

      using Vec = vec::Vectorized<Scalar>;
      Scalar* result_ptr = result_data;
      for (i64 i = 0; i < outer; ++i) {
        for (i64 j = 0; j < ninputs; j++) {
          i64 local_inner = inputs[j].inner_size;
          Scalar* input_ptr = (Scalar*)(inputs[j].data_ptr) + i * local_inner;
          if (local_inner < Vec::size()) {
            #if !defined(_MSC_VER) && !defined(COMPILING_FOR_MIN_SIZE)
            # pragma unroll
            #endif
            for (i64 k = 0; k < local_inner; k++) {
              result_ptr[k] = input_ptr[k];
            }
          } else {
            vec::map(
                [](Vec x) { return x; },
                result_ptr,
                input_ptr,
                local_inner);
          }
          result_ptr += local_inner;
        }
      }
        */
}

pub fn cat_serial_kernel(
        result:  &mut Tensor,
        tensors: TensorList,
        dim:     i64)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, result.scalar_type(), "cat_serial_kernel", [&]() {
        cat_serial_kernel_impl<Scalar>(result, tensors, dim);
      });
        */
}

register_dispatch!{cat_serial_stub, &cat_serial_kernel}
