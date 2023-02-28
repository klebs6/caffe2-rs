crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Cross.h]

lazy_static!{
    /*
    using cross_fn = void(*)(Tensor&, const Tensor&, const Tensor&, const i64 d);
    */
}

declare_dispatch!{cross_fn, cross_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Cross.cpp]

define_dispatch!{cross_stub}

pub fn cross(
        input:     &Tensor,
        other:     &Tensor,
        dimension: Option<i64>) -> Tensor {
    
    todo!();
        /*
            Tensor out = empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      native::cross_out(input, other, dimension, out);
      return out;
        */
}

pub fn cross_out(
        input:     &Tensor,
        other:     &Tensor,
        dimension: Option<i64>,
        out:       &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            auto device_res = input.device().type();
      TORCH_CHECK(device_res == kCPU || device_res == kCUDA, "cross only supports CPU and CUDA devices, out got: ", device_res);
      auto device1 = input.device().type();
      TORCH_CHECK(device1 == kCPU || device1 == kCUDA, "cross only supports CPU and CUDA devices, input got: ", device1);
      auto device2 = other.device().type();
      TORCH_CHECK(device2 == kCPU || device2 == kCUDA, "cross only supports CPU and CUDA devices, other got: ", device2);
      TORCH_CHECK(device_res == device1, "out and input must have the same device type. out: ", device_res, " input: ", device1);
      TORCH_CHECK(device1 == device2, "input and other must have the same device type. input: ", device1, " other: ", device2);
      TORCH_CHECK(!out.is_cuda() || out.get_device() == input.get_device(), "device of out (", input.get_device(), ") must match device of input (", other.get_device(), ")");
      TORCH_CHECK(!input.is_cuda() || input.get_device() == other.get_device(), "device of input (", input.get_device(), ") must match device of other (", other.get_device(), ")");
      TORCH_CHECK(input.dim() == other.dim(), "inconsistent tensors dimensions input: ", input.dim(), " other: ", other.dim());
      TORCH_CHECK(input.sizes() == other.sizes(), "inconsistent tensors sizes input: ", input.sizes(), " other: ", other.sizes());

      i64 dim = -1;
      if(!dimension.has_value()) {
        for(i64 i = 0; i < input.dim(); i++) {
          if(input.size(i) == 3) {
            dim = i;
            break;
          }
        }
        TORCH_CHECK(dim >= 0, "no dimension of size 3 in input");
      } else {
        dim = maybe_wrap_dim(dimension.value(), input.dim());
        TORCH_CHECK(input.size(dim) == 3, "dimension ", dimension.value(), " does not have size 3");
      }

      if (out.sizes() != input.sizes()) {
        out.resize_as_(input);
      }

      cross_stub(device1, out, input, other, dim);
      return out;
        */
}
