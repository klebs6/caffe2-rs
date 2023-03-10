/*!
  | WARNING: this header contains non-inline
  | functions and should be only included
  | from ONE cpp file
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ComplexHelper.h]

/**
  | View tensor with new dtype, storage
  | offset, sizes and strides
  |
  */
#[inline] pub fn view_tensor(
        tensor:  &Tensor,
        dtype:   ScalarType,
        offset:  i64,
        sizes:   &[i32],
        strides: &[i32]) -> Tensor {
    
    todo!();
        /*
            Storage storage = tensor.storage();
      auto key_set = tensor.key_set().remove(DispatchKey::Conjugate);
      auto new_tensor = make_tensor<TensorImpl>(
          TensorImpl::VIEW, move(storage), key_set, scalarTypeToTypeMeta(dtype));
      auto * impl = new_tensor.unsafeGetTensorImpl();
      impl->set_storage_offset(offset);
      impl->set_sizes_and_strides(sizes, strides);
      return new_tensor;
        */
}

#[inline] pub fn compute_stride_for_view_as_real(oldstride: &[i32]) -> DimVector {
    
    todo!();
        /*
            DimVector res(oldstride.size() + 1);
      for(usize i = 0; i < oldstride.size(); i++) {
        res[i] = oldstride[i] * 2;
      }
      res.back() = 1;
      return res;
        */
}

/**
  | expects as input a complex tensor and returns
  | back a tensor with corresponding real dtype
  | containing the complex values in the last two
  | dimensions
  |
  */
pub fn view_as_real(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(!self.is_conj(), "view_as_real doesn't work on unresolved conjugated tensors.  To resolve the conjugate tensor so you can view it as real, use self.resolve_conj(); however, be warned that the resulting tensor will NOT alias the original.");
      return native::_view_as_real_physical(self);
        */
}

pub fn view_as_real_physical(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(self.is_complex(), "view_as_real_physical is only supported for complex tensors");
      auto old_sizes = self.sizes();
      DimVector new_sizes(old_sizes.size() + 1);
      copy(old_sizes.begin(), old_sizes.end(), new_sizes.begin());
      // last dimension will always have two elements containing the real and imag vals
      new_sizes.back() = 2;
      auto new_strides = computeStrideForViewAsReal(self.strides());
      auto new_storage_offset = 2 * self.storage_offset();
      const auto float_type = toValueType(self.scalar_type());
      auto real_tensor = view_tensor(self, float_type, new_storage_offset, new_sizes, new_strides);
      return real_tensor;
        */
}

#[inline] pub fn compute_stride_for_view_as_complex(oldstride: &[i32]) -> DimVector {
    
    todo!();
        /*
            const i64 dim = oldstride.size();
      TORCH_CHECK(oldstride[dim-1] == 1, "Tensor must have a last dimension with stride 1");

      DimVector res(dim - 1);
      for (i64 i = 0; i < res.size(); i++) {
        TORCH_CHECK(oldstride[i] % 2 == 0, "Tensor must have a stride divisible by 2 for all but last dimension");
        res[i] = oldstride[i] / 2;
      }
      return res;
        */
}

/**
  | expects as input a float or double tensor
  | with last dimension of size 2 and returns
  | back a tensor with corresponding complex
  | dtype
  |
  */
pub fn view_as_complex(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
        self.scalar_type() == kFloat || self.scalar_type() == kDouble || self.scalar_type() == kHalf,
        "view_as_complex is only supported for half, float and double tensors, but got a tensor of scalar type: ", self.scalar_type());

      auto old_sizes = self.sizes();
      TORCH_CHECK(old_sizes.size() != 0, "Input tensor must have one or more dimensions");
      TORCH_CHECK(old_sizes[old_sizes.size()-1] == 2, "Tensor must have a last dimension of size 2");
      DimVector new_sizes(old_sizes.begin(), old_sizes.end() - 1);

      const auto new_strides = computeStrideForViewAsComplex(self.strides());
      const auto complex_type = toComplexType(self.scalar_type());

      TORCH_CHECK(self.storage_offset() % 2 == 0, "Tensor must have a storage_offset divisible by 2");
      const auto new_storage_offset = self.storage_offset() / 2;

      return view_tensor(self, complex_type, new_storage_offset, new_sizes, new_strides);
        */
}
