// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/quantized_ops.h]

lazy_static!{
    /*
    using qrelu_fn = void (*)(const Tensor& /*qx*/, Tensor& /*qy*/);

    using qrelu_leaky_fn = void (*)(Tensor& /*out*/, const Tensor& /*qx*/, const Scalar& /*negval_*/);

    using qsigmoid_fn = void (*)(const Tensor& /*qx*/, Tensor& /*qy*/, double output_scale, i64 output_zero_point);
    using qhardsigmoid_fn = void (*)(const Tensor& /*qx*/, Tensor& /*qy*/);
    using qclamp_fn = void (*)(
        const Tensor& /*qx*/,
        const Scalar& min,
        const Scalar& max,
        Tensor& /*qy*/);
    using qclamp_minmax_fn = void (*)(
        const Tensor& /*qx*/,
        const Scalar& /*min or max*/,
        Tensor& /*qy*/);
    using qthreshold_fn = void (*)(
        const Tensor& /*qx*/,
        const Scalar& threshold,
        const Scalar& value,
        Tensor& /*qy*/);
    using qtanh_fn = void (*)(const Tensor& /*qx*/, Tensor& /*qy*/);
    using qelu_fn = void(*)(
        const Tensor& /*qx*/,
        const Scalar& /*alpha*/,
        const Scalar& /*scale*/,
        const Scalar& /*input_scale*/,
        Tensor& /*qy*/);
    using qbinary_fn =
        void (*)(Tensor& /*out*/, const Tensor& /*self*/, const Tensor& /*other*/);
    using qadd_scalar_fn =
        void (*)(Tensor& /*out*/, const Tensor& /*self*/, const Scalar& other /*other*/);
    using qhardswish_fn = void (*)(const Tensor& /*qx*/, Tensor& /*qy*/);
    using qmaxpool_2d_fn = void (*)(
        const Tensor& qx,
        i64 iC, // input/output channels
        i64 iH,
        i64 iW, // input sizes
        i64 oH,
        i64 oW, // output sizes
        i64 kH,
        i64 kW, // kernel size
        i64 sH,
        i64 sW, // strides
        i64 pH,
        i64 pW, // padding
        i64 dH,
        i64 dW, // dilation
        Tensor& qy);
    using qadaptive_avg_pool2d_fn = void (*)(
        const Tensor& qx,
        Tensor& qy,
        i64 b,
        i64 sizeC,
        i64 isizeH,
        i64 isizeW,
        i64 osizeH,
        i64 osizeW,
        i64 istrideB,
        i64 istrideC,
        i64 istrideH,
        i64 istrideW);
    using qadaptive_avg_pool3d_fn = void (*)(
        const Tensor& qx,
        Tensor& qy,
        i64 b,
        i64 sizeC,
        i64 isizeD,
        i64 isizeH,
        i64 isizeW,
        i64 osizeD,
        i64 osizeH,
        i64 osizeW,
        i64 istrideB,
        i64 istrideC,
        i64 istrideD,
        i64 istrideH,
        i64 istrideW);
    using qavg_pool2d_fn = void (*)(
        const Tensor& qx,
        Tensor& qy,
        i64 b,
        i64 nInputPlane,
        i64 inputWidth,
        i64 inputHeight,
        i64 outputWidth,
        i64 outputHeight,
        int kW,
        int kH,
        int dW,
        int dH,
        int padW,
        int padH,
        bool count_include_pad,
        optional<i64> divisor_override);

    using qavg_pool3d_fn = void (*)(
        const Tensor& qx,
        Tensor& qy,
        i64 b,
        i64 nInputPlane,
        i64 inputWidth,
        i64 inputHeight,
        i64 inputDepth,
        i64 outputWidth,
        i64 outputHeight,
        i64 outputDepth,
        int kW,
        int kH,
        int kD,
        int dW,
        int dH,
        int dD,
        int padW,
        int padH,
        int padD,
        bool count_include_pad,
        optional<i64> divisor_override);

    using qupsample_bilinear2d_fn = void (*)(
        Tensor& output,
        const Tensor& input,
        i64 input_height,
        i64 input_width,
        i64 output_height,
        i64 output_width,
        i64 nbatch,
        i64 channels,
        bool align_corners,
        optional<double> scales_h,
        optional<double> scales_w);

    using qcat_nhwc_fn = Tensor (*)(
        const List<Tensor>& qxs,
        i64 dim,
        double scale,
        i64 zero_point);

    using qtopk_fn = void(*)(Tensor&, Tensor&, const Tensor&, i64, i64, bool, bool);

    using qbatch_norm_fn = void(*)(i64, i64, i64, i64, i64, const Tensor&, const Tensor&, const Tensor&, Tensor&);

    using qnormalize_fn = void (*)(
        const Tensor& /* X */,
        const Tensor& /* gamma */,
        const Tensor& /* beta */,
        bool /* affine_per_channel */,
        int /* num_channels */,
        int /* num_groups */,
        i64 /* M */,
        i64 /* N */,
        double /* eps */,
        Tensor* /* Y */);
    */
}

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
