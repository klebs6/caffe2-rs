// # vim: ft=none
/*!
  | Basic functions on sparse tensors
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/sparse/SparseCsrTensor.cpp]

pub fn validate_sparse_csr_tensor_args(
        crow_indices: &Tensor,
        col_indices:  &Tensor,
        values:       &Tensor,
        size:         &[i32])  {
    
    todo!();
        /*
            // Layout Invariants
      TORCH_CHECK(
          col_indices.layout() == kStrided && col_indices.is_contiguous(),
          "expected col_indices to be a strided and contiguous tensor");

      TORCH_CHECK(
          crow_indices.layout() == kStrided && crow_indices.is_contiguous(),
          "expected crow_indices to be a strided and contiguous tensor");

      TORCH_CHECK(
          values.layout() == kStrided && values.is_contiguous(),
          "expected values to be a strided and contiguous tensor");

      // Shape and Strides invariants
      TORCH_CHECK(
          size.size() == 2,
          "size of a CSR tensor must be of length 2, but got: ",
          size.size());
      TORCH_CHECK(
          crow_indices.dim() == 1,
          "crow_indices must have dim=1 but got crow_indices.dim()=",
          crow_indices.dim());
      TORCH_CHECK(
          col_indices.dim() == 1,
          "col_indices must have dim=1 but got col_indices.dim()=",
          col_indices.dim());
      TORCH_CHECK(
          values.dim() == 1,
          "values must have dim=1 but got values.dim()=",
          values.dim());
      // Note, this check also enforces `crow_indices.numel() >= 1`
      TORCH_CHECK(
          crow_indices.numel() == (size[0] + 1),
          "crow_indices.numel() must be size(0) + 1, but got: ",
          crow_indices.numel());
      TORCH_CHECK(
          col_indices.numel() == values.numel(),
          "col_indices and values must have equal sizes, but got col_indices.numel(): ",
          col_indices.numel(),
          ", values.numel(): ",
          values.numel());

      // Indices invariants
      AT_DISPATCH_INDEX_TYPES(crow_indices.scalar_type(), "csr_construct_check", [&] {
        Tensor crow_indices_cpu = crow_indices.to(kCPU);
        auto crow_indices_accessor = crow_indices_cpu.accessor<Index, 1>();
        TORCH_CHECK(
            crow_indices_accessor[0] == 0, "0th value of crow_indices must be 0.");

        TORCH_CHECK(
            crow_indices_accessor[crow_indices.numel() - 1] == col_indices.numel(),
            "last value of crow_indices should be equal to the length of col_indices.");

        for (int i =  1; i <= size[0]; i++) {
          TORCH_CHECK(
              crow_indices_accessor[i - 1] <= crow_indices_accessor[i],
              "at position i = ", i, ", this condition crow_indices[i - 1] <= crow_indices[i] fails");
        }
        if (col_indices.numel() > 0) {
          TORCH_CHECK(0 <= col_indices.min().item<Index>(), "col_indices.min() should be greater or equal to zero");
          TORCH_CHECK(size[1] > col_indices.max().item<Index>(), "size(1) should be greater than col_indices.max()");
        }
      });

      // CSR Type Invariants
      auto crow_indices_type = crow_indices.scalar_type();
      auto col_indices_type = col_indices.scalar_type();
      TORCH_CHECK(
          crow_indices_type == col_indices_type,
          "both crow_indices and col_indices should have the same type.");
      TORCH_CHECK(
          crow_indices_type == kInt || crow_indices_type == kLong,
          "crow_indices and col_indices must be an int32 or int64 type, but got: ",
          crow_indices_type);

      // CSR Device Invariants
      TORCH_CHECK(
          col_indices.get_device() == crow_indices.get_device(),
          "crow_indices and col_indices devices (",
          crow_indices.get_device(),
          ", ",
          col_indices.get_device(),
          ") must match");
      TORCH_CHECK(
          crow_indices.get_device() == values.get_device(),
          "device of crow_indices (",
          crow_indices.get_device(),
          ") must match device of values (",
          values.get_device(),
          ")");
      TORCH_CHECK(
          values.device().type() == kCPU || values.device().type() == kCUDA,
          "device type of values (",
          values.device().type(),
          ") must be CPU or CUDA");
        */
}

/**
  | Construction of CSR tensors.
  |
  */
pub fn new_csr_tensor(options: &TensorOptions) -> SparseCsrTensor {
    
    todo!();
        /*
            // TODO: remove this comment after enabling autograd support for CSR tensor
      // constructor.
      // TORCH_INTERNAL_ASSERT(variable_excluded_from_dispatch());
      TORCH_INTERNAL_ASSERT(options.layout() == kSparseCsr);
      DispatchKey dispatch_key;

      TORCH_CHECK_NOT_IMPLEMENTED(
        options.device().type() == kCPU || options.device().type() == kCUDA,
         "Could not run '", "sparse_csr_tensor", "' from the '", options.device(), "' device.)");

      if (options.device().is_cuda()) {
        dispatch_key = DispatchKey::SparseCsrCUDA;
      } else {
        dispatch_key = DispatchKey::SparseCsrCPU;
      }

      return make_tensor<SparseCsrTensorImpl>(
          DispatchKeySet(dispatch_key), options.dtype());
        */
}

pub fn sparse_csr_tensor_unsafe(
    crow_indices: &Tensor,
    col_indices:  &Tensor,
    values:       &Tensor,
    size:         &[i32],
    dtype:        Option<ScalarType>,
    layout:       Option<Layout>,
    device:       Option<Device>,
    pin_memory:   Option<bool>) -> Tensor {

    todo!();
        /*
            TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      SparseCsrTensor self = new_csr_tensor(options);
      get_sparse_csr_impl(self)->set_member_tensors(crow_indices, col_indices, values, size);
      return self;
        */
}

/**
  | TODO: This constructor should probably
  | use an ATen abstract method in order
  | to make autograd dispatch available
  | for the CSR constructor.
  | 
  | See the relevant note in native_functions.yaml.
  |
  */
pub fn sparse_csr_tensor_a(
        crow_indices: &Tensor,
        col_indices:  &Tensor,
        values:       &Tensor,
        size:         &[i32],
        dtype:        Option<ScalarType>,
        layout:       Option<Layout>,
        device:       Option<Device>,
        pin_memory:   Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

      native::_validate_sparse_csr_tensor_args(crow_indices, col_indices, values, size);

      return native::_sparse_csr_tensor_unsafe(
          crow_indices,
          col_indices,
          values,
          size,
          optTypeMetaToScalarType(options.dtype_opt()),
          options.layout_opt(),
          options.device_opt(),
          options.pinned_memory_opt());
        */
}

pub fn sparse_csr_tensor_b(
    crow_indices: &Tensor,
    col_indices:  &Tensor,
    values:       &Tensor,
    dtype:        Option<ScalarType>,
    layout:       Option<Layout>,
    device:       Option<Device>,
    pin_memory:   Option<bool>) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for TensorOptions]
      TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
      array<i64, 2> size;
      if (col_indices.numel() > 0) {
        AT_DISPATCH_INDEX_TYPES(col_indices.scalar_type(), "csr_construct_check", [&] {
          size[0] = crow_indices.numel() - 1;
          size[1] = col_indices.max().item<Index>() + 1;
        });
      } else {
        size[0] = 0;
        size[1] = 0;
      }

      native::_validate_sparse_csr_tensor_args(crow_indices, col_indices, values, size);

      return native::_sparse_csr_tensor_unsafe(
          crow_indices,
          col_indices,
          values,
          size,
          optTypeMetaToScalarType(options.dtype_opt()),
          options.layout_opt(),
          options.device_opt(),
          options.pinned_memory_opt());
        */
}

/**
  | Access members of CSR tensors.
  |
  */
pub fn nnz_sparse_csr(self_: &SparseCsrTensor) -> i64 {
    
    todo!();
        /*
            return get_sparse_csr_impl(self)->nnz();
        */
}

pub fn values_sparse_csr(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return get_sparse_csr_impl(self)->values().alias();
        */
}

pub fn crow_indices_sparse_csr(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return get_sparse_csr_impl(self)->crow_indices().alias();
        */
}

pub fn col_indices_sparse_csr(self_: &Tensor) -> Tensor {
    
    todo!();
        /*
            return get_sparse_csr_impl(self)->col_indices().alias();
        */
}

pub fn is_same_size_as_sparse_csr(
        self_: &SparseCsrTensor,
        src:   &SparseCsrTensor) -> bool {
    
    todo!();
        /*
            return self.sizes().equals(src.sizes());
        */
}

pub fn resize_as_sparse_csr(
        self_: &SparseCsrTensor,
        src:   &SparseCsrTensor) -> &SparseCsrTensor {
    
    todo!();
        /*
            TORCH_CHECK(
          src.is_sparse_csr() && self.is_sparse_csr(),
          "resize_as_sparse_csr_: layout for self and src must be sparse_csr but got self, src: ",
          self.layout(),
          src.layout());
      if (!_is_same_size_as_sparse_csr(self, src)) {
        get_sparse_csr_impl(self)->resize_as_sparse_csr_tensor_(src);
      }
      return self;
        */
}
