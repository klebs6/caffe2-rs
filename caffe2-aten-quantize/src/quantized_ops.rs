crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/quantized_ops.h]

pub type QreluFn = fn(qx: &Tensor, qy: &mut Tensor) -> c_void;

pub type QreluLeakyFn = fn(
        out:    &mut Tensor,
        qx:     &Tensor,
        negval: &Scalar
) -> c_void;

pub type QsigmoidFn = fn(
        qx:                &Tensor,
        qy:                &mut Tensor,
        output_scale:      f64,
        output_zero_point: i64
) -> c_void;

pub type QhardsigmoidFn = fn(qx: &Tensor, qy: &mut Tensor) -> c_void;

pub type QclampFn = fn(
        qx:  &Tensor,
        min: &Scalar,
        max: &Scalar,
        qy:  &mut Tensor
) -> c_void;

pub type QclampMinmaxFn = fn(
        qx:         &Tensor,
        min_or_max: &Scalar,
        qy:         &mut Tensor
) -> c_void;

pub type QthresholdFn = fn(
        qx:        &Tensor,
        threshold: &Scalar,
        value:     &Scalar,
        qy:        &mut Tensor
) -> c_void;

pub type QtanhFn = fn(qx: &Tensor, qy: &mut Tensor) -> c_void;

pub type QeluFn = fn(
        qx:          &Tensor,
        alpha:       &Scalar,
        scale:       &Scalar,
        input_scale: &Scalar,
        qy:          &mut Tensor
) -> ();

pub type QbinaryFn = fn(
        out:   &mut Tensor,
        self_: &Tensor,
        other: &Tensor
) -> c_void;

pub type QaddScalarFn = fn(
out:    &mut Tensor,
self_:    &Tensor,
other: &Scalar
) -> c_void;


pub type QhardswishFn = fn(qx: &Tensor, qy: &mut Tensor) -> c_void;

pub type Qmaxpool2dFn = fn(
        qx: &Tensor,

        // input_output channels
        ic: i64,
        ih: i64,

        // input sizes
        iw: i64,
        oh: i64,

        // output sizes
        ow: i64,
        kh: i64,

        // kernel size
        kw: i64,
        sh: i64,

        // strides
        sw: i64,
        ph: i64,

        // padding
        pw: i64,
        dh: i64,

        // dilation
        dw: i64,
        qy: &mut Tensor
) -> c_void;

pub type QadaptiveAvgPool2dFn = fn(
        qx:       &Tensor,
        qy:       &mut Tensor,
        b:        i64,
        sizec:    i64,
        isizeh:   i64,
        isizew:   i64,
        osizeh:   i64,
        osizew:   i64,
        istrideb: i64,
        istridec: i64,
        istrideh: i64,
        istridew: i64
) -> c_void;

pub type QadaptiveAvgPool3dFn = fn(
        qx:       &Tensor,
        qy:       &mut Tensor,
        b:        i64,
        sizec:    i64,
        isized:   i64,
        isizeh:   i64,
        isizew:   i64,
        osized:   i64,
        osizeh:   i64,
        osizew:   i64,
        istrideb: i64,
        istridec: i64,
        istrided: i64,
        istrideh: i64,
        istridew: i64
) -> c_void;

pub type QavgPool2dFn = fn(
        qx:                &Tensor,
        qy:                &mut Tensor,
        b:                 i64,
        n_input_plane:     i64,
        input_width:       i64,
        input_height:      i64,
        output_width:      i64,
        output_height:     i64,
        kw:                i32,
        kh:                i32,
        dw:                i32,
        dh:                i32,
        padw:              i32,
        padh:              i32,
        count_include_pad: bool,
        divisor_override:  Option<i64>
) -> c_void;

pub type QavgPool3dFn = fn(
        qx:                &Tensor,
        qy:                &mut Tensor,
        b:                 i64,
        n_input_plane:     i64,
        input_width:       i64,
        input_height:      i64,
        input_depth:       i64,
        output_width:      i64,
        output_height:     i64,
        output_depth:      i64,
        kw:                i32,
        kh:                i32,
        kd:                i32,
        dw:                i32,
        dh:                i32,
        dd:                i32,
        padw:              i32,
        padh:              i32,
        padd:              i32,
        count_include_pad: bool,
        divisor_override:  Option<i64>
) -> c_void;

pub type QupsampleBilinear2dFn = fn(
        output:        &mut Tensor,
        input:         &Tensor,
        input_height:  i64,
        input_width:   i64,
        output_height: i64,
        output_width:  i64,
        nbatch:        i64,
        channels:      i64,
        align_corners: bool,
        scales_h:      Option<f64>,
        scales_w:      Option<f64>
) -> c_void;

pub type QcatNhwcFn = fn(
        qxs:        &List<Tensor>,
        dim:        i64,
        scale:      f64,
        zero_point: i64
) -> Tensor;

pub type QtopkFn = fn(
        _0: &mut Tensor,
        _1: &mut Tensor,
        _2: &Tensor,
        _3: i64,
        _4: i64,
        _5: bool,
        _6: bool
) -> ();

pub type QbatchNormFn = fn(
        _0: i64,
        _1: i64,
        _2: i64,
        _3: i64,
        _4: i64,
        _5: &Tensor,
        _6: &Tensor,
        _7: &Tensor,
        _8: &mut Tensor
) -> ();

pub type QnormalizeFn = fn(
        X:                  &Tensor,
        gamma:              &Tensor,
        beta:               &Tensor,
        affine_per_channel: bool,
        num_channels:       i32,
        num_groups:         i32,
        M:                  i64,
        N:                  i64,
        eps:                f64,
        Y:                  *mut Tensor
) -> c_void;


declare_dispatch!{qadaptive_avg_pool2d_fn, qadaptive_avg_pool2d_nhwc_stub}
declare_dispatch!{qadaptive_avg_pool3d_fn, qadaptive_avg_pool3d_ndhwc_stub}
declare_dispatch!{qadd_scalar_fn, qadd_scalar_relu_stub}
declare_dispatch!{qadd_scalar_fn, qadd_scalar_stub}
declare_dispatch!{qavg_pool2d_fn, qavg_pool2d_nhwc_stub}
declare_dispatch!{qavg_pool3d_fn, qavg_pool3d_nhwc_stub}
declare_dispatch!{qbatch_norm_fn, qbatch_norm_relu_stub}
declare_dispatch!{qbatch_norm_fn, qbatch_norm_stub}
declare_dispatch!{qbinary_fn, qadd_relu_stub}
declare_dispatch!{qbinary_fn, qadd_stub}
declare_dispatch!{qbinary_fn, qmul_relu_stub}
declare_dispatch!{qbinary_fn, qmul_stub}
declare_dispatch!{qcat_nhwc_fn, qcat_nhwc_stub}
declare_dispatch!{qcat_nhwc_fn, qcat_relu_nhwc_stub}
declare_dispatch!{qclamp_fn, qclamp_stub}
declare_dispatch!{qclamp_minmax_fn, qclamp_min_stub}
declare_dispatch!{qclamp_minmax_fn, qclamp_max_stub}
declare_dispatch!{qelu_fn, qelu_stub}
declare_dispatch!{qhardsigmoid_fn, qhardsigmoid_stub}
declare_dispatch!{qhardswish_fn, qhardswish_stub}
declare_dispatch!{qmaxpool_2d_fn, qmaxpool_2d_nhwc_stub}
declare_dispatch!{qnormalize_fn, quantized_normalize_stub}
declare_dispatch!{qrelu_fn, qrelu6_stub}
declare_dispatch!{qrelu_fn, qrelu_stub}
declare_dispatch!{qrelu_leaky_fn, qrelu_leaky_stub}
declare_dispatch!{qsigmoid_fn, qsigmoid_stub}
declare_dispatch!{qtanh_fn, qtanh_stub}
declare_dispatch!{qthreshold_fn, qthreshold_stub}
declare_dispatch!{qtopk_fn, qtopk_stub}
declare_dispatch!{qupsample_bilinear2d_fn, qupsample_bilinear2d_nhwc_stub}
