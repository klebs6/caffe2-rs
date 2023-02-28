// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/pack.h]

/**
  | Legend:
  |
  |  dq: Design-time Quantization
  |
  |  rq: Run-time Quantization
  */
#[inline] pub fn pytorch_pack_q8gemm_wdq(
    nc:       usize,
    kc:       usize,
    nr:       u32,
    np:       u32,
    kr:       u32,
    izp:      u8,
    kzp:      u8,
    k:        *const u8,
    b:        *const i32,
    packed_w: *mut void)  {

    todo!();
        /*
            const i32 boff = (i32)kc * (i32)izp * (i32)kzp;
      for (usize nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
        const usize nr_block_size = min(nc - nr_block_start, nr);
        i32* packed_b = (i32*)packed_w;
        for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          *((i32*)packed_w) = b ? b[nr_block_start + nr_block_offset] + boff : 0.0f;
          packed_w = (void*)((uintptr_t)packed_w + sizeof(i32));
        }
        packed_w =
            (void*)((uintptr_t)packed_w + (nr - nr_block_size) * sizeof(i32));
        for (usize kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
          const usize kr_block_size = min(kc - kr_block_start, kr);
          for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
               nr_block_offset++) {
            i32 ksum = 0;
            for (usize kr_block_offset = 0; kr_block_offset < kr_block_size;
                 kr_block_offset++) {
              const u8 kv =
                  k[(nr_block_start + nr_block_offset) * kc +
                    (kr_block_start + kr_block_offset)];
              ksum += (i32)kv;
              *((u8*)packed_w) = kv;
              packed_w = (void*)((uintptr_t)packed_w + sizeof(u8));
            }
            packed_b[nr_block_offset] -= ksum * (i32)izp;
            packed_w =
                (void*)((uintptr_t)packed_w + (kr - kr_block_size) * sizeof(u8));
          }
          packed_w =
              (void*)((uintptr_t)packed_w + ((nr - nr_block_size) & (np - 1)) * kr * sizeof(u8));
        }
      }
        */
}

/**
  | NB: We use the same packing function for both
  | dynamic quantization and runtime quantization
  | for linear.
  |
  | This means that dynamic mode will suffer some
  | perf because of the branching introduced due to
  | `if(kzp!=0)` however, that should not be too
  | significant.
  */
#[inline] pub fn pytorch_pack_q8gemm_wrq(
    nc:       usize,
    kc:       usize,
    nr:       u32,
    np:       u32,
    kr:       u32,
    k:        *const u8,
    b:        *const i32,
    kzp:      *const u8,
    packed_w: *mut void)  {
    
    todo!();
        /*
            union {
        void* const as_void_ptr;
        u8* as_uint8_ptr;
        i32* as_int32_ptr;
      } packed = {packed_w};

      for (usize nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
        const usize nr_block_size = min(nc - nr_block_start, nr);
        for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          *(packed.as_int32_ptr++) = b ? b[nr_block_start + nr_block_offset] : 0.0f;
        }
        packed.as_int32_ptr += (nr - nr_block_size);
        for (usize kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
          const usize kr_block_size = min(kc - kr_block_start, kr);
          for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
               nr_block_offset++) {
            for (usize kr_block_offset = 0; kr_block_offset < kr_block_size;
                 kr_block_offset++) {
              const u8 kv =
                  k[(nr_block_start + nr_block_offset) * kc +
                    (kr_block_start + kr_block_offset)];
              *(packed.as_uint8_ptr++) = kv;
            }
            // Weights need to be prepacked with the zero points, in their tail space
            // where packed blocks are not multiple of input sizes
            // e.g for ukernels with kr=2 and k is 3 then the second block must be
            // padded with zero point. This is because when subtracting with zero point
            // we just get zero for the padded value, which is what we want.
            if (kzp != 0) {
              for (usize kr_block_offset = 0; kr_block_offset < (kr - kr_block_size);
                   kr_block_offset++) {
                const u8 kv =
                    kzp[(nr_block_start + nr_block_offset)];
                *(packed.as_uint8_ptr++) = kv;
              }
            } else {
              packed.as_uint8_ptr += (kr - kr_block_size);
            }
          }
          if (kzp != 0) {
            // This part fills the packed weights with zero points for output channels
            // when they are not divisble by nr blocking parameter.
            // This is needed because in some kernels, sse2 ones, it relies on this
            // to produce zero as a result of subtracting zero point from weight value.
            usize remaining_nr_blocks = ((nr - nr_block_size) & (np - 1));
            for (usize nr_block_offset = 0; nr_block_offset < remaining_nr_blocks;
                 nr_block_offset++) {
              for (usize kr_block_offset = 0; kr_block_offset < kr;
                   kr_block_offset++) {
                const u8 kv =
                    kzp[(nr_block_start + nr_block_size + nr_block_offset)];
                *(packed.as_uint8_ptr++) = kv;
              }
            }
          } else {
            packed.as_uint8_ptr += ((nr - nr_block_size) & (np - 1)) * kr;
          }
        }
      }
        */
}

#[inline] pub fn pytorch_pack_q8conv_wdq(
    n:        usize,
    ks:       usize,
    kc:       usize,
    nr:       u32,
    kr:       u32,
    izp:      u8,
    kzp:      u8,
    k:        *const u8,
    b:        *const i32,
    packed_w: *mut void)  {

    todo!();
        /*
            const i32 boff = (i32)ks * (i32)kc * (i32)izp * (i32)kzp;
      for (usize nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
        const usize nr_block_size = min(n - nr_block_start, nr);
        i32* packed_b = (i32*)packed_w;
        for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          *((i32*)packed_w) = b ? b[nr_block_start + nr_block_offset] + boff : 0.0f;
          packed_w = (void*)((uintptr_t)packed_w + sizeof(i32));
        }
        packed_w =
            (void*)((uintptr_t)packed_w + (nr - nr_block_size) * sizeof(i32));
        for (usize ki = 0; ki < ks; ki++) {
          for (usize kr_block_start = 0; kr_block_start < kc;
               kr_block_start += kr) {
            const usize kr_block_size = min(kc - kr_block_start, kr);
            for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
                 nr_block_offset++) {
              i32 ksum = 0;
              for (usize kr_block_offset = 0; kr_block_offset < kr_block_size;
                   kr_block_offset++) {
                const u8 kv =
                    k[((nr_block_start + nr_block_offset) * ks + ki) * kc +
                      (kr_block_start + kr_block_offset)];
                ksum += (i32)kv;
                *((u8*)packed_w) = kv;
                packed_w = (void*)((uintptr_t)packed_w + sizeof(u8));
              }
              packed_b[nr_block_offset] -= ksum * (i32)izp;
              packed_w =
                  (void*)((uintptr_t)packed_w + (kr - kr_block_size) * sizeof(u8));
            }
            packed_w =
                (void*)((uintptr_t)packed_w + (nr - nr_block_size) * kr * sizeof(u8));
          }
        }
      }
        */
}

#[inline] pub fn pytorch_pack_q8conv_wrq(
    n:        usize,
    ks:       usize,
    kc:       usize,
    nr:       u32,
    kr:       u32,
    k:        *const u8,
    b:        *const i32,
    kzp:      *const u8,
    packed_w: *mut void)  {

    todo!();
        /*
            union {
        void* const as_void_ptr;
        u8* as_uint8_ptr;
        i32* as_int32_ptr;
      } packed = {packed_w};

      for (usize nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
        const usize nr_block_size = min(n - nr_block_start, nr);
        for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          *(packed.as_int32_ptr++) = b ? b[nr_block_start + nr_block_offset] : 0.0f;
        }
        packed.as_int32_ptr += (nr - nr_block_size);
        for (usize ki = 0; ki < ks; ki++) {
          for (usize kr_block_start = 0; kr_block_start < kc;
               kr_block_start += kr) {
            const usize kr_block_size = min(kc - kr_block_start, kr);
            for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
                 nr_block_offset++) {
              for (usize kr_block_offset = 0; kr_block_offset < kr_block_size;
                   kr_block_offset++) {
                const u8 kv =
                    k[((nr_block_start + nr_block_offset) * ks + ki) * kc +
                      (kr_block_start + kr_block_offset)];
                *(packed.as_uint8_ptr++) = kv;
              }
              // Weights need to be prepacked with the zero points, in their tail space
              // where packed blocks are not multiple of input sizes
              // e.g for ukernels with kr=2 and k is 3 then the second block must be
              // padded with zero point. This is because when subtracting with zero point
              // we just get zero for the padded value, which is what we want.
              if (kzp != 0) {
                for (usize kr_block_offset = 0; kr_block_offset < (kr - kr_block_size);
                     kr_block_offset++) {
                  const u8 kv =
                      kzp[(nr_block_start + nr_block_offset)];
                  *(packed.as_uint8_ptr++) = kv;
                }
              } else {
                packed.as_uint8_ptr += (kr - kr_block_size);
              }
            }
            if (kzp != 0) {
              // This part fills the packed wights with zero points for output channels
              // when they are not divisble by nr blocking parameter.
              // In that case
              for (usize nr_block_offset = 0; nr_block_offset < (nr - nr_block_size);
                   nr_block_offset++) {
                for (usize kr_block_offset = 0; kr_block_offset < kr;
                     kr_block_offset++) {
                  const u8 kv =
                      kzp[(nr_block_start + nr_block_size + nr_block_offset)];
                  *(packed.as_uint8_ptr++) = kv;
                }
              }
            } else {
              packed.as_uint8_ptr += (nr - nr_block_size) * kr;
            }
          }
        }
      }
        */
}

#[inline] pub fn pytorch_pack_q8deconv_wdq(
    n:        usize,
    ks:       usize,
    kc:       usize,
    nr:       u32,
    kr:       u32,
    izp:      u8,
    kzp:      u8,
    k:        *const u8,
    b:        *const i32,
    packed_w: *mut void)  {
    
    todo!();
        /*
            const i32 boff = (i32)ks * (i32)kc * (i32)izp * (i32)kzp;
      for (usize nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
        const usize nr_block_size = min(n - nr_block_start, nr);
        i32* packed_b = (i32*)packed_w;
        for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          *((i32*)packed_w) = b ? b[nr_block_start + nr_block_offset] + boff : 0.0f;
          packed_w = (void*)((uintptr_t)packed_w + sizeof(i32));
        }
        packed_w =
            (void*)((uintptr_t)packed_w + (nr - nr_block_size) * sizeof(i32));
        for (usize ki = 0; ki < ks; ki++) {
          for (usize kr_block_start = 0; kr_block_start < kc;
               kr_block_start += kr) {
            const usize kr_block_size = min(kc - kr_block_start, kr);
            for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
                 nr_block_offset++) {
              i32 ksum = 0;
              for (usize kr_block_offset = 0; kr_block_offset < kr_block_size;
                   kr_block_offset++) {
                const u8 kv =
                    k[((kr_block_start + kr_block_offset) * ks + ki) * n +
                      (nr_block_start + nr_block_offset)];
                ksum += (i32)kv;
                *((u8*)packed_w) = kv;
                packed_w = (void*)((uintptr_t)packed_w + sizeof(u8));
              }
              packed_b[nr_block_offset] -= ksum * (i32)izp;
              packed_w =
                  (void*)((uintptr_t)packed_w + (kr - kr_block_size) * sizeof(u8));
            }
            packed_w =
                (void*)((uintptr_t)packed_w + (nr - nr_block_size) * kr * sizeof(u8));
          }
        }
      }
        */
}


#[inline] pub fn pytorch_pack_q8deconv_wrq(
    n:        usize,
    ks:       usize,
    kc:       usize,
    nr:       u32,
    kr:       u32,
    k:        *const u8,
    b:        *const i32,
    kzp:      *const u8,
    packed_w: *mut void)  {

    todo!();
        /*
            union {
        void* const as_void_ptr;
        u8* as_uint8_ptr;
        i32* as_int32_ptr;
      } packed = {packed_w};

      for (usize nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
        const usize nr_block_size = min(n - nr_block_start, nr);
        for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          *(packed.as_int32_ptr++) = b ? b[nr_block_start + nr_block_offset] : 0.0f;
        }
        packed.as_int32_ptr += (nr - nr_block_size);
        for (usize ki = 0; ki < ks; ki++) {
          for (usize kr_block_start = 0; kr_block_start < kc;
               kr_block_start += kr) {
            const usize kr_block_size = min(kc - kr_block_start, kr);
            for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
                 nr_block_offset++) {
              for (usize kr_block_offset = 0; kr_block_offset < kr_block_size;
                   kr_block_offset++) {
                const u8 kv =
                    k[((kr_block_start + kr_block_offset) * ks + ki) * n +
                      (nr_block_start + nr_block_offset)];
                *(packed.as_uint8_ptr++) = kv;
              }
              // Weights need to be prepacked with the zero points, in their tail space
              // where packed blocks are not multiple of input sizes
              // e.g for ukernels with kr=2 and k is 3 then the second block must be
              // padded with zero point. This is because when subtracting with zero point
              // we just get zero for the padded value, which is what we want.
              if (kzp != 0) {
                for (usize kr_block_offset = 0; kr_block_offset < (kr - kr_block_size);
                     kr_block_offset++) {
                  const u8 kv =
                      kzp[(nr_block_start + nr_block_offset)];
                  *(packed.as_uint8_ptr++) = kv;
                }
              } else {
                packed.as_uint8_ptr += (kr - kr_block_size);
              }
            }
            if (kzp != 0) {
              // This part fills the packed wights with zero points for output channels
              // when they are not divisble by nr blocking parameter.
              // In that case
              for (usize nr_block_offset = 0; nr_block_offset < (nr - nr_block_size);
                   nr_block_offset++) {
                for (usize kr_block_offset = 0; kr_block_offset < kr;
                     kr_block_offset++) {
                  const u8 kv =
                      kzp[(nr_block_start + nr_block_size + nr_block_offset)];
                  *(packed.as_uint8_ptr++) = kv;
                }
              }
            } else {
              packed.as_uint8_ptr += (nr - nr_block_size) * kr;
            }
          }
        }
      }
        */
}

#[inline] pub fn pytorch_pack_q8dw_wdq(
    h:        usize,
    w:        usize,
    c:        usize,
    cr:       usize,
    izp:      u8,
    kzp:      *mut u8,
    k:        *const u8,
    b:        *const i32,
    packed_w: *mut void)  {

    todo!();
        /*
            const i32 boff = (i32)h * (i32)w * (i32)izp;
      for (usize cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
        const usize cr_block_size = min(c - cr_block_start, cr);
        i32* packed_b = (i32*)packed_w;
        for (usize cr_block_offset = 0; cr_block_offset < cr_block_size;
             cr_block_offset++) {
          *((i32*)packed_w) =
            b ?
                b[cr_block_start + cr_block_offset] +
                boff * kzp[cr_block_start + cr_block_offset] : 0.0f;
          packed_w = (void*)((uintptr_t)packed_w + sizeof(i32));
        }
        packed_w =
            (void*)((uintptr_t)packed_w + (cr - cr_block_size) * sizeof(i32));
        for (usize x = 0; x < w; x++) {
          for (usize y = 0; y < h; y++) {
            for (usize cr_block_offset = 0; cr_block_offset < cr_block_size;
                 cr_block_offset++) {
              const u8 kv =
                  k[((cr_block_start + cr_block_offset) * h + y) * w + x];
              packed_b[cr_block_offset] -= (i32)kv * (i32)izp;
              *((u8*)packed_w) = kv;
              packed_w = (void*)((uintptr_t)packed_w + sizeof(u8));
            }
            packed_w =
                (void*)((uintptr_t)packed_w + (cr - cr_block_size) * sizeof(u8));
          }
        }
      }
        */
}

#[inline] pub fn pytorch_pack_q8dw_wrq(
    h:        usize,
    w:        usize,
    c:        usize,
    cr:       usize,
    k:        *const u8,
    b:        *const i32,
    packed_w: *mut void)  {

    todo!();
        /*
            union {
        void* const as_void_ptr;
        u8* as_uint8_ptr;
        i32* as_int32_ptr;
      } packed = {packed_w};

      for (usize cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
        const usize cr_block_size = min(c - cr_block_start, cr);
        for (usize cr_block_offset = 0; cr_block_offset < cr_block_size;
             cr_block_offset++) {
          *(packed.as_int32_ptr++) = b ? b[cr_block_start + cr_block_offset] : 0.0f;
        }
        packed.as_int32_ptr += (cr - cr_block_size);
        for (usize x = 0; x < w; x++) {
          for (usize y = 0; y < h; y++) {
            for (usize cr_block_offset = 0; cr_block_offset < cr_block_size;
                 cr_block_offset++) {
              const u8 kv =
                  k[((cr_block_start + cr_block_offset) * h + y) * w + x];
              *(packed.as_uint8_ptr++) = kv;
            }
            packed.as_uint8_ptr += (cr - cr_block_size);
          }
        }
      }
        */
}

#[inline] pub fn pytorch_pack_q8dw_w_dilation(
    h:              usize,
    w:              usize,
    c:              usize,
    cr:             usize,
    y_start:        usize,
    y_end:          usize,
    x_start:        usize,
    x_end:          usize,
    k:              *const u8,
    b:              *const i32,
    packed_w:       *mut void,
    pytorch_pack_b: bool)  {
    
    todo!();
        /*
            for (usize cr_block_start = 0; cr_block_start < c; cr_block_start += cr) {
        const usize cr_block_size = min(c - cr_block_start, cr);
        if (pytorch_pack_b) {
          for (usize cr_block_offset = 0; cr_block_offset < cr_block_size;
               cr_block_offset++) {
            *((i32*)packed_w) = b ? b[cr_block_start + cr_block_offset] : 0.0f;
            packed_w = (void*)((uintptr_t)packed_w + sizeof(i32));
          }
          packed_w =
              (void*)((uintptr_t)packed_w + (cr - cr_block_size) * sizeof(i32));
        }
        for (usize x = x_start; x < x_end; x++) {
          for (usize y = y_start; y < y_end; y++) {
            for (usize cr_block_offset = 0; cr_block_offset < cr_block_size;
                 cr_block_offset++) {
              *((u8*)packed_w) =
                  k[((cr_block_start + cr_block_offset) * h + y) * w + x];
              packed_w = (void*)((uintptr_t)packed_w + sizeof(u8));
            }
            packed_w =
                (void*)((uintptr_t)packed_w + (cr - cr_block_size) * sizeof(u8));
          }
        }
      }
        */
}

#[inline] pub fn pytorch_pack_swizzle_q8gemm_bdq(
    n:        usize,
    kc:       usize,
    nr:       u32,
    kr:       u32,
    sr:       u32,
    izp:      u8,
    kzp:      u8,
    k:        *const u8,
    b:        *const i32,
    packed_w: *mut void)  {

    todo!();
        /*
            const i32 boff = (i32)kc * (i32)izp * (i32)kzp;
      for (usize nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
        const usize nr_block_size = min(n - nr_block_start, nr);
        i32* packed_b = (i32*)packed_w;
        for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          *((i32*)packed_w) = b ? b[nr_block_start + nr_block_offset] + boff : 0.0f;
          packed_w = (void*)((uintptr_t)packed_w + sizeof(i32));
        }
        packed_w =
            (void*)((uintptr_t)packed_w + (nr - nr_block_size) * sizeof(i32));

        for (usize kr_block_start = 0; kr_block_start < (kc & -sr);
             kr_block_start += kr) {
          for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
               nr_block_offset++) {
            for (usize kr_block_offset = 0; kr_block_offset < kr;
                 kr_block_offset++) {
              const u8 kv =
                  k[(nr_block_start + nr_block_offset) * kc +
                    (kr_block_start & -sr) +
                    ((kr_block_start + nr_block_offset * kr) & (sr - 1)) +
                    kr_block_offset];
              packed_b[nr_block_offset] -= (i32)kv * (i32)izp;
              *((u8*)packed_w) = kv;
              packed_w = (void*)((uintptr_t)packed_w + sizeof(u8));
            }
          }
          packed_w =
              (void*)((uintptr_t)packed_w + (nr - nr_block_size) * kr * sizeof(u8));
        }

        for (usize kr_block_start = (kc & -sr); kr_block_start < kc;
             kr_block_start += kr) {
          const usize kr_block_size = min(kc - kr_block_start, kr);
          for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
               nr_block_offset++) {
            for (usize kr_block_offset = 0; kr_block_offset < kr_block_size;
                 kr_block_offset++) {
              const u8 kv =
                  k[(nr_block_start + nr_block_offset) * kc +
                    (kr_block_start + kr_block_offset)];
              packed_b[nr_block_offset] -= (i32)kv * (i32)izp;
              *((u8*)packed_w) = kv;
              packed_w = (void*)((uintptr_t)packed_w + sizeof(u8));
            }
            packed_w =
                (void*)((uintptr_t)packed_w + (kr - kr_block_size) * sizeof(u8));
          }
          packed_w =
              (void*)((uintptr_t)packed_w + (nr - nr_block_size) * kr * sizeof(u8));
        }
      }
        */
}

#[inline] pub fn pytorch_pack_swizzle_q8gemm_brq(
    n:        usize,
    kc:       usize,
    nr:       u32,
    kr:       u32,
    sr:       u32,
    k:        *const u8,
    b:        *const i32,
    packed_w: *mut void)  {

    todo!();
    /*
            union {
        void* const as_void_ptr;
        u8* as_uint8_ptr;
        i32* as_int32_ptr;
      } packed = {packed_w};

      for (usize nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
        const usize nr_block_size = min(n - nr_block_start, nr);
        for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          *(packed.as_int32_ptr++) = b ? b[nr_block_start + nr_block_offset] : 0.0f;
        }

        packed.as_int32_ptr += (nr - nr_block_size);

        for (usize kr_block_start = 0; kr_block_start < (kc & -sr);
             kr_block_start += kr) {
          for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
               nr_block_offset++) {
            for (usize kr_block_offset = 0; kr_block_offset < kr;
                 kr_block_offset++) {
              const u8 kv =
                  k[(nr_block_start + nr_block_offset) * kc +
                    (kr_block_start & -sr) +
                    ((kr_block_start + nr_block_offset * kr) & (sr - 1)) +
                    kr_block_offset];
              *(packed.as_uint8_ptr++) = kv;
            }
          }
          packed.as_uint8_ptr += (nr - nr_block_size) * kr;
        }

        for (usize kr_block_start = (kc & -sr); kr_block_start < kc;
             kr_block_start += kr) {
          const usize kr_block_size = min(kc - kr_block_start, kr);
          for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
               nr_block_offset++) {
            for (usize kr_block_offset = 0; kr_block_offset < kr_block_size;
                 kr_block_offset++) {
              const u8 kv =
                  k[(nr_block_start + nr_block_offset) * kc +
                    (kr_block_start + kr_block_offset)];
              *(packed.as_uint8_ptr++) = kv;
            }
            packed.as_uint8_ptr += (kr - kr_block_size);
          }
          packed.as_uint8_ptr += (nr - nr_block_size) * kr;
        }
      }
        */
}

#[inline] pub fn pytorch_pack_hgemm_w(
    nc:       usize,
    kc:       usize,
    nr:       usize,
    kr:       usize,
    k:        *const u16,
    b:        *const u16,
    packed_w: *mut u16)  {
    
    todo!();
        /*
            for (usize nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
        const usize nr_block_size = min(nc - nr_block_start, nr);
        for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          *packed_w++ = b ? b[nr_block_start + nr_block_offset] : 0.0f;
        }
        packed_w += nr - nr_block_size;
        for (usize kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
          const usize kr_block_size = min(kc - kr_block_start, kr);
          for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
               nr_block_offset++) {
            for (usize kr_block_offset = 0; kr_block_offset < kr_block_size;
                 kr_block_offset++) {
              *packed_w++ =
                  k[(nr_block_start + nr_block_offset) * kc +
                    (kr_block_start + kr_block_offset)];
            }
            packed_w += kr - kr_block_size;
          }
          packed_w += (nr - nr_block_size) * kr;
        }
      }
        */
}

#[inline] pub fn pytorch_pack_sgemm_w(
    nc:       usize,
    kc:       usize,
    nr:       usize,
    kr:       usize,
    k:        *const f32,
    b:        *const f32,
    packed_w: *mut f32)  {
    
    todo!();
        /*
            for (usize nr_block_start = 0; nr_block_start < nc; nr_block_start += nr) {
        const usize nr_block_size = min(nc - nr_block_start, nr);
        for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          *packed_w++ = b ? b[nr_block_start + nr_block_offset] : 0.0f;
        }
        packed_w += nr - nr_block_size;
        for (usize kr_block_start = 0; kr_block_start < kc; kr_block_start += kr) {
          const usize kr_block_size = min(kc - kr_block_start, kr);
          for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
               nr_block_offset++) {
            for (usize kr_block_offset = 0; kr_block_offset < kr_block_size;
                 kr_block_offset++) {
              *packed_w++ =
                  k[(nr_block_start + nr_block_offset) * kc +
                    (kr_block_start + kr_block_offset)];
            }
            packed_w += kr - kr_block_size;
          }
          packed_w += (nr - nr_block_size) * kr;
        }
      }
        */
}


#[inline] pub fn pytorch_pack_sconv_w(
    n:        usize,
    ks:       usize,
    kc:       usize,
    nr:       usize,
    kr:       usize,
    k:        *const f32,
    b:        *const f32,
    packed_w: *mut f32)  {

    todo!();
        /*
            for (usize nr_block_start = 0; nr_block_start < n; nr_block_start += nr) {
        const usize nr_block_size = min(n - nr_block_start, nr);
        for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
             nr_block_offset++) {
          *packed_w++ = b ? b[nr_block_start + nr_block_offset] : 0.0f;
        }
        packed_w += nr - nr_block_size;
        for (usize ki = 0; ki < ks; ki++) {
          for (usize kr_block_start = 0; kr_block_start < kc;
               kr_block_start += kr) {
            const usize kr_block_size = min(kc - kr_block_start, kr);
            for (usize nr_block_offset = 0; nr_block_offset < nr_block_size;
                 nr_block_offset++) {
              for (usize kr_block_offset = 0; kr_block_offset < kr_block_size;
                   kr_block_offset++) {
                *packed_w++ =
                    k[((nr_block_start + nr_block_offset) * ks + ki) * kc +
                      (kr_block_start + kr_block_offset)];
              }
              packed_w += kr - kr_block_size;
            }
            packed_w += (nr - nr_block_size) * kr;
          }
        }
      }
        */
}

#[cfg(PYTORCH_QNNPACK_RUNTIME_QUANTIZATION)]
lazy_static!{
    /*
    #define pytorch_pack_q8gemm_w pytorch_pack_q8gemm_wrq
    #define pytorch_pack_q8conv_w pytorch_pack_q8conv_wrq
    #define pytorch_pack_q8deconv_w pytorch_pack_q8deconv_wrq
    #define pytorch_pack_q8dw_w pytorch_pack_q8dw_wrq
    #define pytorch_pack_swizzle_q8gemm_b pytorch_pack_swizzle_q8gemm_brq
    */
}

#[cfg(not(PYTORCH_QNNPACK_RUNTIME_QUANTIZATION))]
lazy_static!{
    /*
    #define pytorch_pack_q8gemm_w pytorch_pack_q8gemm_wdq
    #define pytorch_pack_q8conv_w pytorch_pack_q8conv_wdq
    #define pytorch_pack_q8deconv_w pytorch_pack_q8deconv_wdq
    #define pytorch_pack_q8dw_w pytorch_pack_q8dw_wdq
    #define pytorch_pack_swizzle_q8gemm_b pytorch_pack_swizzle_q8gemm_bdq
    */
}
