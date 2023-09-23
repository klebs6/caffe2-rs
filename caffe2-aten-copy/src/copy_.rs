crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Copy.h]

lazy_static!{
    /*
    using copy_fn = void (*)(TensorIterator&, bool non_blocking);
    */
}

declare_dispatch!{copy_fn, copy_stub}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/Copy.cpp]

pub fn copy_transpose_valid(
        self_: &Tensor,
        src:   &Tensor) -> bool {
    
    todo!();
        /*
            const int MIN_SZ = 60 * 60;
      return self.is_contiguous() && src.numel() != 0 && src.dim() == 2 &&
          src.stride(0) == 1 && src.stride(1) == src.size(0) &&
          self.scalar_type() == src.scalar_type() &&
          self.numel() >= MIN_SZ;
        */
}

/**
  | special case copy where tensor is contiguous
  | and src is a transposed matrix
  |
  | This can be generalized to most copies, but
  | it's trickier
  |
  */
pub fn copy_same_type_transpose(
        self_: &mut Tensor,
        src:   &Tensor)  {
    
    todo!();
        /*
      i64 BLOCK_SZ;
      if (self.scalar_type() == kByte) {
        BLOCK_SZ = 120;
      } else {
        BLOCK_SZ = 60;
      }
      Tensor buf = empty({BLOCK_SZ, BLOCK_SZ}, self.options());

      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kHalf, kBool, kBFloat16, self.scalar_type(), "copy_", [&] {
        Scalar* sp = src.data_ptr<Scalar>();
        Scalar* rp = self.data_ptr<Scalar>();
        Scalar* bp = buf.data_ptr<Scalar>();

        i64 NR = src.size(0);
        i64 NC = src.size(1);
        for (i64 R = 0; R < NR; R += BLOCK_SZ) {
          for (i64 C = 0; C < NC; C += BLOCK_SZ) {
            Scalar* spo = sp + R + C * NR;
            Scalar* rpo = rp + C + R * NC;

            int nr = min(NR - R, BLOCK_SZ);
            int nc = min(NC - C, BLOCK_SZ);

            // 1. copy columns from src to buf
            for (int c = 0; c < nc; c++) {
              memcpy(bp + c * BLOCK_SZ, spo + c * NR, nr * sizeof(Scalar));
            }

            // 2. transpose buf in place
            int rc_max = max(nr, nc);
            int rc_min = min(nr, nc);
            for (int r = 0; r < rc_max; r++) {
              int end = min(r, rc_min);
              for (int c = 0; c < end; c++) {
                Scalar tmp = bp[r + BLOCK_SZ * c];
                bp[r + BLOCK_SZ * c] = bp[r * BLOCK_SZ + c];
                bp[r * BLOCK_SZ + c] = tmp;
              }
            }

            // 3. copy rows from buf to dst
            for (int r = 0; r < nr; r++) {
              memcpy(rpo + r * NC, bp + r * BLOCK_SZ, nc * sizeof(Scalar));
            }
          }
        }
      });
        */
}

/**
  | Devices directly supported by this copy
  | implementation.
  |
  | Other device types (e.g. XLA) may be supported
  | by overriding copy_ and _copy_from.
  |
  */
pub fn is_supported_device(device: Device) -> bool {
    
    todo!();
        /*
            DeviceType device_type = device.type();
      return device_type == kCPU || device_type == kCUDA || device_type == kHIP || device_type == kVulkan || device_type == kMetal;
        */
}

pub fn copy_impl<'a>(
        self_:        &mut Tensor,
        src:          &Tensor,
        non_blocking: bool) -> &'a mut Tensor {
    
    todo!();
        /*
            // TODO: this should be handled during dispatch, but that's missing...
      TORCH_CHECK(self.defined(), "self is undefined");
      TORCH_CHECK(src.defined(), "src is undefined");

      // FBGeMM kernel support exists only for the following case,
      // 1. Memory Format for source and destination tensors is contiguous.
      // 2. Device for both the source and destination tensor is CPU.
      // 3. dtype conversion between FP32->FP16 and FP16->FP32.
      #ifdef USE_FBGEMM
        if (((self.dtype() == kFloat && src.dtype() == kHalf) ||
             (self.dtype() == kHalf && src.dtype() == kFloat)) &&
            (self.device().is_cpu() && src.device().is_cpu()) &&
            !self.is_sparse() && !src.is_sparse() &&
            ((self.is_contiguous() && src.is_contiguous()) ||
             (self.is_non_overlapping_and_dense() && self.strides() == src.strides()))) {
          if (src.dtype() == kFloat && self.dtype() == kHalf) {
            auto* output_ptr =
                reinterpret_cast<fbgemm::float16*>(self.data_ptr<Half>());
            if (self.numel() < internal::GRAIN_SIZE) {
              fbgemm::FloatToFloat16_simd(src.data_ptr<float>(), output_ptr, self.numel());
            } else {
              parallel_for(
                  0,
                  self.numel(),
                  internal::GRAIN_SIZE,
                  [&](i64 begin, i64 end) {
                    fbgemm::FloatToFloat16_simd(
                        src.data_ptr<float>() + begin,
                        output_ptr + begin,
                      end - begin);
                  });
            }
          } else {
            auto in_data = reinterpret_cast<fbgemm::float16*>(
                src.data_ptr<Half>());
            auto* output_ptr = self.data_ptr<float>();
            if (self.numel() < internal::GRAIN_SIZE) {
              fbgemm::Float16ToFloat_simd(in_data, output_ptr, self.numel());
            } else {
              parallel_for(
                  0,
                  self.numel(),
                  internal::GRAIN_SIZE,
                  [&](i64 begin, i64 end) {
                    fbgemm::Float16ToFloat_simd(
                        in_data + begin, output_ptr + begin, end - begin);
                  });
            }
          }
          return self;
        }
      #endif

      if (self.is_sparse() && src.is_sparse()) {
        return copy_sparse_to_sparse_(self, src, non_blocking);
      } else if (self.is_sparse() || src.is_sparse()) {
        AT_ERROR("copy_() between dense and sparse Tensors is not implemented! Found self type = ",
                 self.toString(), " and src type = ", src.toString());
      }

      if (self.is_same(src)) {
        return self;
      }

      // Copies into meta self are OK and just ignored (similar to inplace)
      if (self.is_meta()) {
        // TODO: need to see if there is extra error checking needed
        return self;
      }

      if (src.is_meta()) {
        TORCH_CHECK_NOT_IMPLEMENTED(false, "Cannot copy out of meta tensor; no data!")
      }

      // Re-dispatch copies when either src or self device not implemented here (e.g. XLA).
      // _copy_from has a proper device dispatch setup.
      // This includes:
      //   cpu_tensor.copy_(xla_tensor) => xla_tensor._copy_from(cpu_tensor)
      //   xla_tensor.copy_(cpu_tensor) => cpu_tensor._copy_from(xla_tensor)
      // Both the _copy_from calls above will be dispatched to XLA's _copy_from kernels.
      if (!is_supported_device(src.device()) || !is_supported_device(self.device())) {
        _copy_from(src, self, non_blocking);
        return self;
      }

      if (self.is_quantized() && !src.is_quantized()) {
        return quantized_copy_from_float_cpu_(self, src);
      }

      if (self.is_quantized() && src.is_quantized()) {
        TORCH_CHECK(self.qscheme() == src.qscheme(),
                    "Quantized Copy only works with same qscheme");
        TORCH_CHECK(self.scalar_type() == src.scalar_type());
        set_quantizer_(self, src.quantizer());
      }

      if (!self.is_quantized() && src.is_quantized()) {
        TORCH_CHECK(false, "Copying from quantized Tensor to non-quantized Tensor is not allowed, please use dequantize to get a float Tensor from a quantized Tensor");
      }

      if (self.device().type() == kVulkan || src.device().type() == kVulkan) {
      #ifdef USE_VULKAN_API
        return vulkan::ops::copy_(self, src);
      #else
        return vulkan::vulkan_copy_(self, src);
      #endif
      }

      if (self.device().type() == kMetal || src.device().type() == kMetal) {
        return metal::metal_copy_(self, src);
      }

      auto iter = TensorIteratorConfig()
        .add_output(self)
        .add_input(src)
        .resize_outputs(false)
        .check_all_same_dtype(false)
        .check_all_same_device(false)
        .build();

      if (iter.numel() == 0) {
        return self;
      }

      DeviceType device_type = iter.device_type(0);
      if (iter.device_type(1) == kCUDA) {
        device_type = kCUDA;
      } else if (iter.device_type(1) == kHIP) {
        device_type = kHIP;
      }

      // TODO: if we need to, we can also enable this path for quantized tensor
      if (device_type == kCPU && copy_transpose_valid(self, src) && !self.is_quantized()) {
        copy_same_type_transpose_(self, src);
        return self;
      }

      if(!self.is_complex() && src.is_complex()) {
        TORCH_WARN_ONCE("Casting complex values to real discards the imaginary part");
      }
      copy_stub(device_type, iter, non_blocking);
      return self;
        */
}

pub fn copy_<'a>(
        self_:        &mut Tensor,
        src:          &Tensor,
        non_blocking: bool) -> &'a mut Tensor {
    
    todo!();
        /*
            auto maybe_outnames = namedinference::compute_broadcast_outnames(self, src);
      {
        NoNamesGuard guard;
        copy_impl(self, src, non_blocking);
      }
      namedinference::propagate_names_if_nonempty(self, maybe_outnames);
      return self;
        */
}

define_dispatch!{copy_stub}
