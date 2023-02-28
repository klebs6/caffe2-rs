/*!
  | Functions that fill Tensors with constants.
  | Implementations are in Fill.cpp.
  |
  */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Fill.h]

declare_dispatch!{
    fn(_0: &mut TensorIterator, _1: &Scalar) -> (),
    fill_stub
}

pub fn fill_out_a(
        self_: &mut Tensor,
        value: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
        
        */
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Fill.cpp]

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fill ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub fn fill_out_b(
        self_: &mut Tensor,
        value: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            if (self.is_quantized()) {
        Tensor out = ones(self.sizes()).to(kFloat) * value;
        out = out.to(self.device());
        // Trust the `copy_` to handle the quantization and the boundary chacks.
        self.copy_(out);
        return self;
      }
      if (self.device() == kCPU && self.numel() == 1) {
        return scalar_fill(self, value);
      }
      auto iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)  // Fill is idempotent, so overlap is okay
        .check_all_same_dtype(false)
        .add_output(self)
        .resize_outputs(false)
        .build();
      fill_stub(iter.device_type(), iter, value);
      return self;
        */
}

pub fn fill_a(
        self_: &mut Tensor,
        value: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return fill_out(self, value);
        */
}

pub fn fill_b(
        self_: &mut Tensor,
        value: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(), " dimensions.");
      return fill_out(self, value.item());
        */
}

pub fn fill_meta_a(
        self_: &mut Tensor,
        value: &Scalar) -> &mut Tensor {
    
    todo!();
        /*
            return self;
        */
}

pub fn fill_meta_b(
        self_: &mut Tensor,
        value: &Tensor) -> &mut Tensor {
    
    todo!();
        /*
            TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(), " dimensions.");
      return self;
        */
}

define_dispatch!{fill_stub}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fill_diagonal ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub fn fill_diagonal(
        self_:      &mut Tensor,
        fill_value: &Scalar,
        wrap:       bool) -> &mut Tensor {
    
    todo!();
        /*
            i64 nDims = self.dim();
      TORCH_CHECK(nDims >= 2, "dimensions must larger than 1");

      i64 height = self.size(0);
      i64 width = self.size(1);

      if (nDims > 2) {
        i64 dim1 = height;
        for (i64 i = 1; i < nDims; i++) {
          if (self.size(i) != dim1) {
            AT_ERROR("all dimensions of input must be of equal length");
          }
        }
      }

      i64 storage_offset = self.storage_offset();
      vector<i64> sizes;
      vector<i64> strides;
      i64 size = min(height, width);

      i64 stride = 0;
      for (i64 i = 0; i < nDims; i++) {
        stride += self.stride(i);
      }
      strides.push_back(stride);
      sizes.push_back(size);

      auto main_diag = self.as_strided(sizes, strides, storage_offset);
      main_diag.fill_(fill_value);

      if (wrap && nDims == 2 && height > width + 1) {
        vector<i64> wrap_sizes;

        i64 step = width + 1;
        i64 wrap_size = ((self.numel() + step - 1) / step) - size;
        wrap_sizes.push_back(wrap_size);

        i64 offset = self.stride(0) * (width + 1);

        auto wrap_diag = self.as_strided(wrap_sizes, strides, storage_offset + offset);
        wrap_diag.fill_(fill_value);
      }

      return self;
        */
}

pub fn zero_cpu(
        self_:     &mut Tensor,
        nelements: i64) -> &mut Tensor {
    
    todo!();
        /*
            void* ptr = self.data_ptr();
      if (nullptr == ptr) {
        return self.fill_(0);
      }
      i64 size_bytes = nelements * self.dtype().itemsize();
      if (size_bytes > 0) {
        memset(ptr, 0, size_bytes);
      }
      return self;
        */
}

pub fn zero(self_: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            i64 nelements = multiply_integers(self.sizes());
      if (self.device() == kCPU &&
          self.is_non_overlapping_and_dense() &&
          nelements < internal::GRAIN_SIZE) {
        return zero_cpu_(self, nelements);
      }
      return self.fill_(0);
        */
}

pub fn zero_meta(self_: &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            return self;
        */
}
