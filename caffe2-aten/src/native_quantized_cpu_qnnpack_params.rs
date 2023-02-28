crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/params.h]

pub struct PyTorchQnnpFp16ClampingParams {
    scale: u16,
    max:   u16,
    min:   u16,
}

pub struct PyTorchQnnpFp32ClampingParams {
    max: f32,
    min: f32,
}

pub enum PyTorchQnnpFp32RequantizationParams {
    Scalar {
        scales:                *mut f32,
        output_zero_point:     u8,
        output_max:            u8,
        output_min:            u8,
        min_less_zero_point:   f32,
        max_less_zero_point:   f32,
        magic:                 f32,
        magic_less_zero_point: i32,
    },
    Neon {
        scales:                *mut f32,
        max:                   f32,
        min:                   f32,
        magic:                 f32,
        magic_less_zero_point: i32,
    },
    NeonV8 {
        scales:     *mut f32,
        zero_point: i16,
        max:        u8,
        min:        u8,
    },
    Sse2 {
        scales:     Align16<*mut f32>,
        zero_point: Align16<[i16; 8]>,
        max:        Align16<[u8; 16]>,
        min:        Align16<[u8; 16]>,
    },
    PSimd {
        scales:                Align16<*mut f32>,
        min_less_zero_point:   Align16<[f32; 4]>,
        max_less_zero_point:   Align16<[f32; 4]>,
        magic:                 Align16<[f32; 4]>,
        magic_less_zero_point: Align16<[i32; 4]>,
    }
}

pub enum PytorchQnnpPreciseRequantizationParams {
    Scalar {
        multiplier:          u32,
        rounding_lo:         u32,
        rounding_hi:         u32,
        shift_less_32:       u32,
        min_less_zero_point: i32,
        max_less_zero_point: i32,
        zero_point:          i32,
    },
    Neon {
        multiplier:  i32,
        right_shift: i32,
        zero_point:  i16,
        max:         u8,
        min:         u8,
    },
    Sse2 {
        multiplier: Align16<[u32; 4]>,
        rounding:   Align16<[u64; 2]>,
        shift:      Align16<[u32; 4]>,
        zero_point: Align16<[i16; 8]>,
        max:        Align16<[u8; 16]>,
        min:        Align16<[u8; 16]>,
    },
}

pub enum PyTorchQnnpQ31RequantizationParams {

    Scalar {
        multiplier:          i32,
        remainder_mask:      i32,
        remainder_threshold: i32,
        shift:               u32,
        min_less_zero_point: i32,
        max_less_zero_point: i32,
        zero_point:          i32,
    },

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    Neon {
        multiplier:  i32,
        right_shift: i32,
        zero_point:  i16,
        max:         u8,
        min:         u8,
    },

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2 {
        multiplier:          Align16<[u32; 4]>,
        rounding:            Align16<[u64; 2]>,
        remainder_mask:      Align16<[i32; 4]>,
        remainder_threshold: Align16<[i32; 4]>,
        shift:               Align16<[u64; 2]>,
        zero_point:          Align16<[i16; 8]>,
        max:                 Align16<[u8; 16]>,
        min:                 Align16<[u8; 16]>,
    },
}

pub enum PyTorchQnnpConvQuantizationParams {

    Scalar {
        kernel_zero_points:         *const u8,
        input_zero_point:           i32,
        requantization_scales:      *const f32,
        output_min_less_zero_point: i32,
        output_max_less_zero_point: i32,
        output_zero_point:          i32,
    },

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    Neon {
        kernel_zero_points:    *const u8,
        input_zero_point:      i16,
        requantization_scales: *const f32,
        output_zero_point:     i16,
        output_max:            u8,
        output_min:            u8,

        /**
          | Following four are for nearest-ties-to-even
          | rounding in aarch32. This saves some
          | instructions needed otherwise.
          |
          */
        vfmax:                 f32,
        vfmin:                 f32,
        vfmagic:               f32,
        vimagic:               i32,
    },

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2 {
        kernel_zero_points:    Align16<*const u8>,
        input_zero_point:      Align16<[i16; 8]>,
        requantization_scales: Align16<*mut f32>,
        output_zero_point:     Align16<[i16; 8]>,
        output_max:            Align16<[u8; 16]>,
        output_min:            Align16<[u8; 16]>,
    },
}

pub struct PyTorchQnnpConvDynamicQuantizationParams {
    input_zero_point:   i16,
    kernel_zero_points: *const u8,
    multipliers:        *const f32,
}

pub enum PytorchQnnpRequantizationParams {
    Precise(PytorchQnnpPreciseRequantizationParams),
    Fp32(PyTorchQnnpFp32RequantizationParams),
    Q31(PyTorchQnnpQ31RequantizationParams),
}

pub enum PyTorchQnnpAddQuantizationParams {

    Scalar {
        zero_point_product:  i32,
        a_multiplier:        u32,
        b_multiplier:        u32,
        shift:               u32,
        remainder_mask:      i32,
        remainder_threshold: i32,
        y_zero_point:        i32,
        y_max:               i32,
        y_min:               i32,
    },

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    Neon {
        a_zero_point: u8,
        b_zero_point: u8,
        y_zero_point: i16,
        a_multiplier: i32,
        b_multiplier: i32,
        right_shift:  i32,
        y_max:        u8,
        y_min:        u8,
    },

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2 {
        zero_point_product:  Align16<[i32; 4]>,
        a_multiplier_lo:     Align16<[u16; 8]>,
        a_multiplier_hi:     Align16<[u16; 8]>,
        b_multiplier_lo:     Align16<[u16; 8]>,
        b_multiplier_hi:     Align16<[u16; 8]>,
        remainder_mask:      Align16<[i32; 4]>,
        remainder_threshold: Align16<[i32; 4]>,
        y_zero_point:        Align16<[i16; 8]>,
        y_max:               Align16<[u8; 16]>,
        y_min:               Align16<[u8; 16]>,
        shift:               u32,
        a_multiplier:        u32,
        b_multiplier:        u32,
    },
}

pub enum PyTorchQnnpAvgPoolQuantizationParams {

    Scalar {
        bias:              i32,
        scale:             f32,
        output_zero_point: i32,
        output_max:        u8,
        output_min:        u8,
    },

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    Neon {
        bias:              i32,
        scale:             f32,
        output_zero_point: i16,
        output_max:        u8,
        output_min:        u8,

        /**
          | Following four are for nearest-ties-to-even
          | rounding in aarch32. This saves some
          | instructions needed otherwise.
          |
          */
        vfmax:             f32,
        vfmin:             f32,
        vfmagic:           f32,
        vimagic:           i32,
    },

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2 {
        bias:              Align16<[i32; 4]>,
        scale:             Align16<[f32; 4]>,
        output_zero_point: Align16<[i16; 8]>,
        output_max:        Align16<[u8; 16]>,
        output_min:        Align16<[u8; 16]>,
    },
}

pub enum PyTorchQnnpU8ClampingParams {

    Scalar {
        output_max: i32,
        output_min: i32,
    },

    #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
    Neon {
        output_max: u8,
        output_min: u8,
    },

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2 {
        output_max: Align16<[u8; 16]>,
        output_min: Align16<[u8; 16]>,
    },
}

pub type PytorchQ8gemmUkernelFunction = fn(
        mr:                   Usize,
        nr:                   Usize,
        k:                    Usize,
        a:                    *const u8,
        a_stride:             Usize,
        w:                    *const c_void,
        c:                    *mut u8,
        c_stride:             Usize,
        output_channel_index: Usize,
        quantization_params:  *const PyTorchQnnpConvQuantizationParams
) -> c_void;

/**
  | Q8 GEMM kernel with support for dynamic
  | quantization.
  | 
  | The w parameter designates weights,
  | and is to be passed on to this kernel exactly
  | as returned by the pack function. The
  | initial bias portion of this buffer
  | will be ignored.
  | 
  | The bias parameter, expects max(nr,
  | 8) floating-point biases. Technically
  | the kernels only need nr biases from
  | the buffer pointed to by this parameter,
  | but end up reading at most 8 to keep the
  | logic simple and fast. Consequently,
  | make sure this parameter has enough
  | storage for 8 floating point numbers
  | to avoid triggering out of bound errors.
  | The remaining 8 - nr biases, if any, will
  | be unused.
  | 
  | quantization_params contains the
  | quantization parameters, namely input,
  | and kernel zero points, and the multiplier.
  | The multiplier is expected to be equal
  | to input_scale * kernel_scale.
  |
  */
pub type PytorchQ8gemmDqUkernelFunction = fn(
        mr:                   Usize,
        nr:                   Usize,
        k:                    Usize,
        a:                    *const u8,
        a_stride:             Usize,
        w:                    *const c_void,
        bias:                 *const f32,
        c:                    *mut f32,
        c_stride:             Usize,
        output_channel_index: Usize,
        quantization_params:  *const PyTorchQnnpConvDynamicQuantizationParams
) -> c_void;

pub type PytorchQ8gemmDqSparseUkernelFunction = fn(
        mr:                   Usize,
        nr:                   Usize,
        a:                    *const u8,
        a_stride:             Usize,
        packed_w:             *const u8,
        w_row_ptr:            *const u32,
        w_block_ids_ptr:      *const u32,
        bias:                 *const f32,
        c:                    *mut f32,
        c_stride:             Usize,
        output_channel_index: Usize,
        quantization_params:  *const PyTorchQnnpConvDynamicQuantizationParams
) -> c_void;

pub type PytorchQ8gemmDqSparsePackedAUkernelFunction = fn(
        mr:                   Usize,
        nr:                   Usize,
        a_packed:             *const u8,
        packed_w:             *const u8,
        w_row_ptr:            *const u32,
        w_block_ids_ptr:      *const u32,
        bias:                 *const f32,
        c:                    *mut f32,
        c_stride:             Usize,
        output_channel_index: Usize,
        quantization_params:  *const PyTorchQnnpConvDynamicQuantizationParams
) -> c_void;

pub type PytorchQ8gemmSparsePackAUkernelFunction = fn(
        mr:       Usize,
        K:        Usize,
        a:        *const u8,
        a_stride: Usize,
        a_packed: *mut u8
) -> c_void;

pub type PytorchQ8convUkernelFunction = fn(
        mr:                   Usize,
        nr:                   Usize,
        kc:                   Usize,
        ks:                   Usize,
        a:                    *const *const u8,
        w:                    *const c_void,
        c:                    *mut u8,
        c_stride:             Usize,
        output_channel_index: Usize,
        quantization_params:  *const PyTorchQnnpConvQuantizationParams
) -> c_void;

pub type PytorchQ8gemmXzpUkernelFunction = fn(
        mr:                    Usize,
        nr:                    Usize,
        k:                     Usize,
        a:                     *const u8,
        a_stride:              Usize,
        a_sum:                 *const i32,
        w:                     *const c_void,
        c:                     *mut u8,
        c_stride:              Usize,
        requantization_params: *const PyTorchQnnpQ31RequantizationParams
) -> c_void;

pub type PytorchQ8sumRowsUkernelFunction = fn(
        a:          *const u8,
        m:          Usize,
        k:          Usize,
        stride:     Usize,
        multiplier: i32,
        sums:       *mut i32
) -> c_void;

pub type PytorchXzipcUkernelFunction = fn(
        n: Usize,
        x: *const c_void,
        y: *mut c_void
) -> c_void;

pub type PytorchXzipvUkernelFunction = fn(
        n: Usize,
        m: Usize,
        x: *const c_void,
        y: *mut c_void
) -> c_void;

pub type PytorchX8lutUkernelFunction = fn(
        n: Usize,
        x: *const u8,
        t: *const u8,
        y: *mut u8
) -> c_void;

pub type PytorchSgemmUkernelFunction = fn(
        mr:              Usize,
        nr:              Usize,
        k:               Usize,
        a:               *const f32,
        a_stride:        Usize,
        w:               *const f32,
        c:               *mut f32,
        c_stride:        Usize,
        clamping_params: *const PyTorchQnnpFp32ClampingParams
) -> c_void;

pub type PytorchSconvUkernelFunction = fn(
        mr:              Usize,
        nr:              Usize,
        kc:              Usize,
        ks:              Usize,
        a:               *const *const f32,
        w:               *const f32,
        c:               *mut f32,
        c_stride:        Usize,
        clamping_params: *const PyTorchQnnpFp32ClampingParams
) -> c_void;

pub type PytorchHgemmUkernelFunction = fn(
        mr:              Usize,
        nr:              Usize,
        k:               Usize,
        a:               *const c_void,
        a_stride:        Usize,
        w:               *const c_void,
        c:               *mut c_void,
        c_stride:        Usize,
        clamping_params: *const PyTorchQnnpFp16ClampingParams
) -> c_void;

pub type PytorchQ8dwconvUpUkernelFunction = fn(
        channels:            Usize,
        output_width:        Usize,
        input:               *const *const u8,
        weights:             *const c_void,
        output:              *mut u8,
        input_stride:        Usize,
        output_increment:    Usize,
        quantization_params: *const PyTorchQnnpConvQuantizationParams
) -> c_void;

pub type PytorchQ8dwconvMpUkernelFunction = fn(
        channels:            Usize,
        output_width:        Usize,
        input:               *const *const u8,
        weights:             *const c_void,
        buffer:              *mut i32,
        output:              *mut u8,
        input_stride:        Usize,
        output_increment:    Usize,
        quantization_params: *const PyTorchQnnpConvQuantizationParams
) -> c_void;

pub type PytorchQ8gavgpoolUpUkernelFunction = fn(
        m:                   Usize,
        n:                   Usize,
        x:                   *const u8,
        x_stride:            Usize,
        zero:                *const u8,
        y:                   *mut u8,
        quantization_params: *const PyTorchQnnpAvgPoolQuantizationParams
) -> c_void;

pub type PytorchQ8gavgpoolMpUkernelFunction = fn(
        m:                   Usize,
        n:                   Usize,
        x:                   *const u8,
        x_stride:            Usize,
        zero:                *const u8,
        buffer:              *mut i32,
        y:                   *mut u8,
        quantization_params: *const PyTorchQnnpAvgPoolQuantizationParams
) -> c_void;

pub type PytorchQ8avgpoolUpUkernelFunction = fn(
        n:                   Usize,
        ks:                  Usize,
        kc:                  Usize,
        x:                   *const *const u8,
        zero:                *const u8,
        y:                   *mut u8,
        x_increment:         Usize,
        y_increment:         Usize,
        quantization_params: *const PyTorchQnnpAvgPoolQuantizationParams
) -> c_void;

pub type PytorchQ8avgpoolMpUkernelFunction = fn(
        n:                   Usize,
        ks:                  Usize,
        kc:                  Usize,
        x:                   *const *const u8,
        zero:                *const u8,
        buffer:              *mut i32,
        y:                   *mut u8,
        x_increment:         Usize,
        y_increment:         Usize,
        quantization_params: *const PyTorchQnnpAvgPoolQuantizationParams
) -> c_void;

pub type PytorchU8maxpoolUkernelFunction = fn(
        n:           Usize,
        ks:          Usize,
        kc:          Usize,
        x:           *const *const u8,
        y:           *mut u8,
        x_increment: Usize,
        y_increment: Usize,
        params:      *const PyTorchQnnpU8ClampingParams
) -> c_void;

pub type PytorchU8clampUkernelFunction = fn(
        n:      Usize,
        x:      *const u8,
        y:      *mut u8,
        params: *const PyTorchQnnpU8ClampingParams
) -> c_void;

pub type PytorchU8rmaxUkernelFunction = fn(n: Usize, x: *const u8) -> u8;

pub type PytorchU8lut32normUkernelFunction = fn(
        n: Usize,
        x: *const u8,
        t: *const u32,
        y: *mut u8
) -> c_void;

pub type PytorchQ8vaddUkernelFunction = fn(
        n:                   Usize,
        a:                   *const u8,
        b:                   *const u8,
        y:                   *mut u8,
        quantization_params: *const PyTorchQnnpAddQuantizationParams
) -> c_void;

pub struct PytorchQ8convParameters {
    gemm:    PyTorchQ8GemmUKernelFunction,
    conv:    PyTorchQ8ConvUKernelFunction,
    gemm_dq: PyTorchQ8GemmDqUKernelFunction,
    mr:      u8,
    nr:      u8,
    kr:      u8,
}

pub struct PytorchQ8gemmSparseParameters {
    gemm_dq:             PyTorchQ8GemmDqSparseUKernelFunction,
    packeda_gemm_dq:     PyTorchQ8GemmDqSparsePAckedAUKernelFunction,
    packa:               PyTorchQ8GemmSparsePackAUKernelFunction,
    mr:                  u8,
    nr:                  u8,
    kr:                  u8,
    log2_mr:             u8,
    log2_row_block_size: u8,
    row_block_size:      u32,
    col_block_size:      u32,
}

pub struct PytorchQ8convXzpParameters {
    gemm:       PyTorchQ8GemmXzpUKernelFunction,

    /**
      | no conv ukernel
      |
      */
    mr:         u8,

    nr:         u8,
    kr:         u8,
    kc:         u8,
    kthreshold: Usize,
}

pub struct PytorchQ8dwconvUpParameters {
    updw:             PyTorchQ8DwConvUpUKernelFunction,
    updw_per_channel: PyTorchQ8DwConvUpUKernelFunction,
    cr:               u8,
}

pub struct PytorchQ8dwconvMpParameters {
    mpdw:             PyTorchQ8DwConvMpUKernelFunction,
    mpdw_per_channel: PyTorchQ8DwConvMpUKernelFunction,
    cr:               u8,
}

pub struct PytorchQ8sumRowsParameters {
    sum_rows: PyTorchQ8SumRowsUKernelFunction,
    m:        u32,
}

pub struct PytorchQ8gavgpoolParameters {
    ltnr:      PyTorchQ8GAvgPoolUpUKernelFunction,
    genr_lemr: PyTorchQ8GAvgPoolUpUKernelFunction,
    genr_gtmr: PyTorchQ8GAvgPoolMpUKernelFunction,
    mr:        u8,
    nr:        u8,
}

pub struct PytorchQ8avgpoolParameters {
    ltkr:      PyTorchQ8AvgPoolUpUKernelFunction,
    gekr_lemr: PyTorchQ8AvgPoolUpUKernelFunction,
    gekr_gtmr: PyTorchQ8AvgPoolMpUKernelFunction,
    mr:        u8,
    qr:        u8,
    kr:        u8,
}

pub struct PytorchU8maxpoolParameters {
    ltkr: PyTorchU8MaxPoolUKernelFunction,
    gekr: PyTorchU8MaxPoolUKernelFunction,
    mr:   u8,
    qr:   u8,
    kr:   u8,
}

pub struct PytorchX8zipParameters {
    x2: PyTorchXZipcUKernelFunction,
    x3: PyTorchXZipcUKernelFunction,
    x4: PyTorchXZipcUKernelFunction,
    xm: PyTorchXZipvUKernelFunction,
}

pub struct PytorchQnnpParameters {
    q8conv:             PytorchQ8convParameters,
    q8gemm_sparse_c1x4: PytorchQ8gemmSparseParameters,
    q8gemm_sparse_c8x1: PytorchQ8gemmSparseParameters,
    q8conv_xzp:         PytorchQ8convXzpParameters,
    q8dw9:              PytorchQ8dwconvUpParameters,
    q8dw25:             PytorchQ8dwconvMpParameters,
    q8sum_rows:         PytorchQ8sumRowsParameters,
    q8vadd:             PyTorchQ8VAddUKernelFunction,
    q8gavgpool:         PytorchQ8gavgpoolParameters,
    q8avgpool:          PytorchQ8avgpoolParameters,
    u8maxpool:          PytorchU8maxpoolParameters,
    u8lut_32norm:       PyTorchU8Lut32NormUKernelFunction,
    u8clamp:            PyTorchU8ClampUKernelFunction,
    u8rmax:             PyTorchU8RMaxUKernelFunction,
    x8zip:              PytorchX8zipParameters,
    x8lut:              PyTorchX8LutUKernelFunction,
    initialized:        bool,
}
