// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/xnnpack/RegisterOpContextClass.cpp]

lazy_static!{
    /*
    TORCH_LIBRARY(xnnpack, m) {
      m.class_<LinearOpContext>("LinearOpContext")
        .def_pickle(
            [](const intrusive_ptr<LinearOpContext>& op_context)
                -> SerializationTypeLinearPrePack { // __getstate__
              return op_context->unpack();
            },
            [](SerializationTypeLinearPrePack state)
                -> intrusive_ptr<LinearOpContext> { // __setstate__
              return createLinearClampPrePackOpContext(
                  move(get<0>(state)),
                  move(get<1>(state)),
                  move(get<2>(state)),
                  move(get<3>(state)));
            });

      m.class_<Conv2dOpContext>("Conv2dOpContext")
        .def_pickle(
            [](const intrusive_ptr<Conv2dOpContext>& op_context)
                -> SerializationTypeConv2dPrePack { // __getstate__
              return op_context->unpack();
            },
            [](SerializationTypeConv2dPrePack state)
                -> intrusive_ptr<Conv2dOpContext> { // __setstate__
              return createConv2dClampPrePackOpContext(
                  move(get<0>(state)),
                  move(get<1>(state)),
                  move(get<2>(state)),
                  move(get<3>(state)),
                  move(get<4>(state)),
                  move(get<5>(state)),
                  move(get<6>(state)),
                  move(get<7>(state)));
            });

      m.class_<TransposeConv2dOpContext>("TransposeConv2dOpContext")
        .def_pickle(
            [](const intrusive_ptr<TransposeConv2dOpContext>& op_context)
                -> SerializationTypeTransposeConv2dPrePack { // __getstate__
              return op_context->unpack();
            },
            [](SerializationTypeTransposeConv2dPrePack state)
                -> intrusive_ptr<TransposeConv2dOpContext> { // __setstate__
              return createConv2dTransposeClampPrePackOpContext(
                  move(get<0>(state)),
                  move(get<1>(state)),
                  move(get<2>(state)),
                  move(get<3>(state)),
                  move(get<4>(state)),
                  move(get<5>(state)),
                  move(get<6>(state)),
                  move(get<7>(state)),
                  move(get<8>(state)));
            });

    }
    */
}

lazy_static!{
    /*
    TORCH_LIBRARY(prepacked, m) {
      m.def(TORCH_SELECTIVE_SCHEMA("prepacked::linear_clamp_prepack(Tensor W, Tensor? B=None, Scalar? output_min=None, Scalar? output_max=None) -> __torch__.torch.classes.xnnpack.LinearOpContext"));
      m.def(TORCH_SELECTIVE_SCHEMA("prepacked::linear_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.LinearOpContext W_prepack) -> Tensor Y"));
      m.def(TORCH_SELECTIVE_SCHEMA("prepacked::conv2d_clamp_prepack(Tensor W, Tensor? B, int[2] stride, int[2] padding, int[2] dilation, int groups, Scalar? output_min=None, Scalar? output_max=None) -> __torch__.torch.classes.xnnpack.Conv2dOpContext"));
      m.def(TORCH_SELECTIVE_SCHEMA("prepacked::conv2d_transpose_clamp_prepack(Tensor W, Tensor? B, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation, int groups, Scalar? output_min=None, Scalar? output_max=None) -> __torch__.torch.classes.xnnpack.TransposeConv2dOpContext"));
      m.def(TORCH_SELECTIVE_SCHEMA("prepacked::conv2d_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.Conv2dOpContext W_prepack) -> Tensor Y"));
      m.def(TORCH_SELECTIVE_SCHEMA("prepacked::conv2d_transpose_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.TransposeConv2dOpContext W_prepack) -> Tensor Y"));
    }

    TORCH_LIBRARY_IMPL(prepacked, CPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("prepacked::linear_clamp_prepack"), TORCH_FN(createLinearClampPrePackOpContext));
      m.impl(TORCH_SELECTIVE_NAME("prepacked::linear_clamp_run"), TORCH_FN(internal::linear::linear_clamp_run));
      m.impl(TORCH_SELECTIVE_NAME("prepacked::conv2d_clamp_prepack"), TORCH_FN(createConv2dClampPrePackOpContext));
      m.impl(TORCH_SELECTIVE_NAME("prepacked::conv2d_transpose_clamp_prepack"), TORCH_FN(createConv2dTransposeClampPrePackOpContext));
      m.impl(TORCH_SELECTIVE_NAME("prepacked::conv2d_clamp_run"), TORCH_FN(internal::convolution2d::conv2d_clamp_run));
      m.impl(TORCH_SELECTIVE_NAME("prepacked::conv2d_transpose_clamp_run"), TORCH_FN(internal::convolution2d::conv2d_transpose_clamp_run));
    }
    */
}
