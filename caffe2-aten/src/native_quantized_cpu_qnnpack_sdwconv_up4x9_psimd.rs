// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/sdwconv/up4x9-psimd.c]

pub fn pytorch_sdwconv_ukernel_up4x9_psimd(
    channels:         usize,
    output_width:     usize,
    input:            *const *const f32,
    weights:          *const f32,
    output:           *mut f32,
    input_stride:     usize,
    output_increment: usize,
    clamping_params:  [PyTorchQnnpFp32ClampingParams; 1])  {

    todo!();
    /*
       const psimd_f32 vmax = psimd_splat_f32(clamping_params->max);
      const psimd_f32 vmin = psimd_splat_f32(clamping_params->min);
      do {
        const float* i0 = input[0];
        const float* i1 = input[1];
        const float* i2 = input[2];
        const float* i3 = input[3];
        const float* i4 = input[4];
        const float* i5 = input[5];
        const float* i6 = input[6];
        const float* i7 = input[7];
        const float* i8 = input[8];

        input = (const float**)((uintptr_t)input + input_stride);

        usize c = channels;
        const float* w = weights;
        for (; c >= 4; c -= 4) {
          psimd_f32 vacc = psimd_load_f32(w);

          const psimd_f32 vi0 = psimd_load_f32(i0);
          i0 += 4;
          const psimd_f32 vk0 = psimd_load_f32(w + 8);
          vacc += vi0 * vk0;

          const psimd_f32 vi1 = psimd_load_f32(i1);
          i1 += 4;
          const psimd_f32 vk1 = psimd_load_f32(w + 12);
          psimd_f32 vacc2 = vi1 * vk1;

          const psimd_f32 vi2 = psimd_load_f32(i2);
          i2 += 4;
          const psimd_f32 vk2 = psimd_load_f32(w + 16);
          vacc += vi2 * vk2;

          const psimd_f32 vi3 = psimd_load_f32(i3);
          i3 += 4;
          const psimd_f32 vk3 = psimd_load_f32(w + 20);
          vacc2 += vi3 * vk3;

          const psimd_f32 vi4 = psimd_load_f32(i4);
          i4 += 4;
          const psimd_f32 vk4 = psimd_load_f32(w + 24);
          vacc += vi4 * vk4;

          const psimd_f32 vi5 = psimd_load_f32(i5);
          i5 += 4;
          const psimd_f32 vk5 = psimd_load_f32(w + 28);
          vacc2 += vi5 * vk5;

          const psimd_f32 vi6 = psimd_load_f32(i6);
          i6 += 4;
          const psimd_f32 vk6 = psimd_load_f32(w + 32);
          vacc += vi6 * vk6;

          const psimd_f32 vi7 = psimd_load_f32(i7);
          i7 += 4;
          const psimd_f32 vk7 = psimd_load_f32(w + 36);
          vacc2 += vi7 * vk7;

          const psimd_f32 vi8 = psimd_load_f32(i8);
          i8 += 4;
          const psimd_f32 vk8 = psimd_load_f32(w + 40);
          vacc += vi8 * vk8;

          vacc += vacc2;

          vacc = psimd_min_f32(vacc, vmax);
          vacc = psimd_max_f32(vacc, vmin);

          psimd_store_f32(output, vacc);
          w += 44;
        }
        if (c != 0) {
          psimd_f32 vacc = psimd_load_f32(w);
          c *= sizeof(float);

          i0 = (const float*)((uintptr_t)i0 - c);
          const psimd_f32 vi0 = psimd_load_f32(i0);
          const psimd_f32 vk0 = psimd_load_f32(w + 8);
          vacc += vi0 * vk0;

          i1 = (const float*)((uintptr_t)i1 - c);
          const psimd_f32 vi1 = psimd_load_f32(i1);
          const psimd_f32 vk1 = psimd_load_f32(w + 12);
          psimd_f32 vacc2 = vi1 * vk1;

          i2 = (const float*)((uintptr_t)i2 - c);
          const psimd_f32 vi2 = psimd_load_f32(i2);
          const psimd_f32 vk2 = psimd_load_f32(w + 16);
          vacc += vi2 * vk2;

          i3 = (const float*)((uintptr_t)i3 - c);
          const psimd_f32 vi3 = psimd_load_f32(i3);
          const psimd_f32 vk3 = psimd_load_f32(w + 20);
          vacc2 += vi3 * vk3;

          i4 = (const float*)((uintptr_t)i4 - c);
          const psimd_f32 vi4 = psimd_load_f32(i4);
          const psimd_f32 vk4 = psimd_load_f32(w + 24);
          vacc += vi4 * vk4;

          i5 = (const float*)((uintptr_t)i5 - c);
          const psimd_f32 vi5 = psimd_load_f32(i5);
          const psimd_f32 vk5 = psimd_load_f32(w + 28);
          vacc2 += vi5 * vk5;

          i6 = (const float*)((uintptr_t)i6 - c);
          const psimd_f32 vi6 = psimd_load_f32(i6);
          const psimd_f32 vk6 = psimd_load_f32(w + 32);
          vacc += vi6 * vk6;

          i7 = (const float*)((uintptr_t)i7 - c);
          const psimd_f32 vi7 = psimd_load_f32(i7);
          const psimd_f32 vk7 = psimd_load_f32(w + 36);
          vacc2 += vi7 * vk7;

          i8 = (const float*)((uintptr_t)i8 - c);
          const psimd_f32 vi8 = psimd_load_f32(i8);
          const psimd_f32 vk8 = psimd_load_f32(w + 40);
          vacc += vi8 * vk8;

          vacc += vacc2;

          vacc = psimd_min_f32(vacc, vmax);
          vacc = psimd_max_f32(vacc, vmin);

          output = (float*)((uintptr_t)output - c);
          psimd_store_f32(output, vacc);
        }

        output = (float*)((uintptr_t)output + output_increment);
      } while (--output_width != 0);
        */
}
