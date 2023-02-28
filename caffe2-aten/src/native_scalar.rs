crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Scalar.cpp]

pub fn item(self_: &Tensor) -> Scalar {
    
    todo!();
        /*
            i64 numel = self.numel();
      TORCH_CHECK(numel == 1, "a Tensor with ", numel, " elements cannot be converted to Scalar");
      if (self.is_sparse()) {
        if (self._nnz() == 0) return Scalar(0);
        if (self.is_coalesced()) return _local_scalar_dense(self._values());
        return _local_scalar_dense(self._values().sum());
      } else if (self.is_quantized()) {
        return self.dequantize().item();
      } else {
        return _local_scalar_dense(self);
      }
        */
}

pub fn local_scalar_dense_cpu(self_: &Tensor) -> Scalar {
    
    todo!();
        /*
            Scalar r;
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, self.scalar_type(), "_local_scalar_dense_cpu", [&] {
            Scalar value = *self.data_ptr<Scalar>();
            r = Scalar(value);
          });
      return r;
        */
}
