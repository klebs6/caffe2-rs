// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/sconv/6x8-psimd.c]

pub fn pytorch_sconv_ukernel_6x8_psimd(
    mr:              usize,
    nr:              usize,
    kc:              usize,
    ks:              usize,
    a:               *const *const f32,
    w:               *const f32,
    c:               *mut f32,
    c_stride:        usize,
    clamping_params: [PyTorchQnnpFp32ClampingParams; 1])  {

    todo!();
        /*
            psimd_f32 vacc0x0123 = psimd_load_f32(w);
      w += 4;
      psimd_f32 vacc0x4567 = psimd_load_f32(w);
      w += 4;
      psimd_f32 vacc1x0123 = vacc0x0123;
      psimd_f32 vacc1x4567 = vacc0x4567;
      psimd_f32 vacc2x0123 = vacc0x0123;
      psimd_f32 vacc2x4567 = vacc0x4567;
      psimd_f32 vacc3x0123 = vacc0x0123;
      psimd_f32 vacc3x4567 = vacc0x4567;
      psimd_f32 vacc4x0123 = vacc0x0123;
      psimd_f32 vacc4x4567 = vacc0x4567;
      psimd_f32 vacc5x0123 = vacc0x0123;
      psimd_f32 vacc5x4567 = vacc0x4567;

      do {
        const float* restrict a0 = *a++;
        const float* restrict a1 = *a++;
        const float* restrict a2 = *a++;
        const float* restrict a3 = *a++;
        const float* restrict a4 = *a++;
        const float* restrict a5 = *a++;

        usize k = kc;
        do {
          const psimd_f32 va0 = psimd_splat_f32(*a0);
          a0 += 1;
          const psimd_f32 va1 = psimd_splat_f32(*a1);
          a1 += 1;
          const psimd_f32 va2 = psimd_splat_f32(*a2);
          a2 += 1;
          const psimd_f32 va3 = psimd_splat_f32(*a3);
          a3 += 1;
          const psimd_f32 va4 = psimd_splat_f32(*a4);
          a4 += 1;
          const psimd_f32 va5 = psimd_splat_f32(*a5);
          a5 += 1;

          const psimd_f32 vb0123 = psimd_load_f32(w);
          w += 4;
          const psimd_f32 vb4567 = psimd_load_f32(w);
          w += 4;

          vacc0x0123 += vb0123 * va0;
          vacc0x4567 += vb4567 * va0;
          vacc1x0123 += vb0123 * va1;
          vacc1x4567 += vb4567 * va1;
          vacc2x0123 += vb0123 * va2;
          vacc2x4567 += vb4567 * va2;
          vacc3x0123 += vb0123 * va3;
          vacc3x4567 += vb4567 * va3;
          vacc4x0123 += vb0123 * va4;
          vacc4x4567 += vb4567 * va4;
          vacc5x0123 += vb0123 * va5;
          vacc5x4567 += vb4567 * va5;
        } while (--k != 0);
      } while (--ks != 0);

      const psimd_f32 vmax = psimd_splat_f32(clamping_params->max);
      vacc0x0123 = psimd_min_f32(vacc0x0123, vmax);
      vacc0x4567 = psimd_min_f32(vacc0x4567, vmax);
      vacc1x0123 = psimd_min_f32(vacc1x0123, vmax);
      vacc1x4567 = psimd_min_f32(vacc1x4567, vmax);
      vacc2x0123 = psimd_min_f32(vacc2x0123, vmax);
      vacc2x4567 = psimd_min_f32(vacc2x4567, vmax);
      vacc3x0123 = psimd_min_f32(vacc3x0123, vmax);
      vacc3x4567 = psimd_min_f32(vacc3x4567, vmax);
      vacc4x0123 = psimd_min_f32(vacc4x0123, vmax);
      vacc4x4567 = psimd_min_f32(vacc4x4567, vmax);
      vacc5x0123 = psimd_min_f32(vacc5x0123, vmax);
      vacc5x4567 = psimd_min_f32(vacc5x4567, vmax);

      const psimd_f32 vmin = psimd_splat_f32(clamping_params->min);
      vacc0x0123 = psimd_max_f32(vacc0x0123, vmin);
      vacc0x4567 = psimd_max_f32(vacc0x4567, vmin);
      vacc1x0123 = psimd_max_f32(vacc1x0123, vmin);
      vacc1x4567 = psimd_max_f32(vacc1x4567, vmin);
      vacc2x0123 = psimd_max_f32(vacc2x0123, vmin);
      vacc2x4567 = psimd_max_f32(vacc2x4567, vmin);
      vacc3x0123 = psimd_max_f32(vacc3x0123, vmin);
      vacc3x4567 = psimd_max_f32(vacc3x4567, vmin);
      vacc4x0123 = psimd_max_f32(vacc4x0123, vmin);
      vacc4x4567 = psimd_max_f32(vacc4x4567, vmin);
      vacc5x0123 = psimd_max_f32(vacc5x0123, vmin);
      vacc5x4567 = psimd_max_f32(vacc5x4567, vmin);

      float* c0 = c;
      float* c1 = (float*)((uintptr_t)c0 + c_stride);
      if (mr < 2) {
        c1 = c0;
      }
      float* c2 = (float*)((uintptr_t)c1 + c_stride);
      if (mr <= 2) {
        c2 = c1;
      }
      float* c3 = (float*)((uintptr_t)c2 + c_stride);
      if (mr < 4) {
        c3 = c2;
      }
      float* c4 = (float*)((uintptr_t)c3 + c_stride);
      if (mr <= 4) {
        c4 = c3;
      }
      float* c5 = (float*)((uintptr_t)c4 + c_stride);
      if (mr != 6) {
        c5 = c4;
      }
      if (nr == 8) {
        psimd_store_f32(c0, vacc0x0123);
        c0 += 4;
        psimd_store_f32(c1, vacc1x0123);
        c1 += 4;
        psimd_store_f32(c2, vacc2x0123);
        c2 += 4;
        psimd_store_f32(c3, vacc3x0123);
        c3 += 4;
        psimd_store_f32(c4, vacc4x0123);
        c4 += 4;
        psimd_store_f32(c5, vacc5x0123);
        c5 += 4;

        psimd_store_f32(c0, vacc0x4567);
        psimd_store_f32(c1, vacc1x4567);
        psimd_store_f32(c2, vacc2x4567);
        psimd_store_f32(c3, vacc3x4567);
        psimd_store_f32(c4, vacc4x4567);
        psimd_store_f32(c5, vacc5x4567);
      } else {
        if (nr >= 4) {
          psimd_store_f32(c0, vacc0x0123);
          c0 += 4;
          psimd_store_f32(c1, vacc1x0123);
          c1 += 4;
          psimd_store_f32(c2, vacc2x0123);
          c2 += 4;
          psimd_store_f32(c3, vacc3x0123);
          c3 += 4;
          psimd_store_f32(c4, vacc4x0123);
          c4 += 4;
          psimd_store_f32(c5, vacc5x0123);
          c5 += 4;
          vacc0x0123 = vacc0x4567;
          vacc1x0123 = vacc1x4567;
          vacc2x0123 = vacc2x4567;
          vacc3x0123 = vacc3x4567;
          vacc4x0123 = vacc4x4567;
          vacc5x0123 = vacc5x4567;
          nr -= 4;
        }
        if (nr >= 2) {
          psimd_store2_f32(c0, vacc0x0123);
          c0 += 2;
          psimd_store2_f32(c1, vacc1x0123);
          c1 += 2;
          psimd_store2_f32(c2, vacc2x0123);
          c2 += 2;
          psimd_store2_f32(c3, vacc3x0123);
          c3 += 2;
          psimd_store2_f32(c4, vacc4x0123);
          c4 += 2;
          psimd_store2_f32(c5, vacc5x0123);
          c5 += 2;
          vacc0x0123 = psimd_concat_hi_f32(vacc0x0123, vacc0x0123);
          vacc1x0123 = psimd_concat_hi_f32(vacc1x0123, vacc1x0123);
          vacc2x0123 = psimd_concat_hi_f32(vacc2x0123, vacc2x0123);
          vacc3x0123 = psimd_concat_hi_f32(vacc3x0123, vacc3x0123);
          vacc4x0123 = psimd_concat_hi_f32(vacc4x0123, vacc4x0123);
          vacc5x0123 = psimd_concat_hi_f32(vacc5x0123, vacc5x0123);
          nr -= 2;
        }
        if (nr != 0) {
          psimd_store1_f32(c0, vacc0x0123);
          psimd_store1_f32(c1, vacc1x0123);
          psimd_store1_f32(c2, vacc2x0123);
          psimd_store1_f32(c3, vacc3x0123);
          psimd_store1_f32(c4, vacc4x0123);
          psimd_store1_f32(c5, vacc5x0123);
        }
      }
        */
}
