// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/ops/Register.cpp]

lazy_static!{
    /*
    TORCH_LIBRARY(vulkan, m) {
      m.class_<Conv2dOpContext>("Conv2dOpContext")
          .def_pickle(
              // __getstate__
              [](const intrusive_ptr<Conv2dOpContext>& context) {
                return context->unpack();
              },
              // __setstate__
              [](Conv2dOpContext::State state) {
                return conv2d_clamp_prepack(
                    move(get<0>(state)),
                    move(get<1>(state)),
                    move(get<2>(state)),
                    move(get<3>(state)),
                    move(get<4>(state)),
                    move(get<5>(state)),
                    move(get<6>(state)),
                    move(get<7>(state)));
              });
      m.class_<LinearOpContext>("LinearOpContext")
          .def_pickle(
              // __getstate__
              [](const intrusive_ptr<LinearOpContext>& context) {
                return context->unpack();
              },
              // __setstate__
              [](LinearOpContext::State state) {
                return linear_prepack(
                    move(get<0>(state)), move(get<1>(state)));
              });
    }
    */
}

lazy_static!{
    /*
    TORCH_LIBRARY(vulkan_prepack, m) {
      m.def(
          "conv2d_clamp_prepack(Tensor W, Tensor? B, int[2] stride, "
          "int[2] padding, int[2] dilation, int groups, "
          "Scalar? output_min=None, Scalar? output_max=None) "
          "-> __torch__.torch.classes.vulkan.Conv2dOpContext");
      m.def(
          "conv2d_clamp_run(Tensor X, "
          "__torch__.torch.classes.vulkan.Conv2dOpContext W_prepack) -> Tensor Y");
      m.def(
          "linear_prepack(Tensor W, Tensor? B) "
          "-> __torch__.torch.classes.vulkan.LinearOpContext");
      m.def(
          "linear_run(Tensor X, "
          "__torch__.torch.classes.vulkan.LinearOpContext BW_prepack) -> Tensor Y");
    }
    */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(vulkan_prepack, CPU, m) {
      m.impl("conv2d_clamp_prepack", TORCH_FN(conv2d_clamp_prepack));
      m.impl("linear_prepack", TORCH_FN(linear_prepack));
    }
    */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(vulkan_prepack, Vulkan, m) {
      m.impl("conv2d_clamp_run", TORCH_FN(conv2d_clamp_run));
      m.impl("linear_run", TORCH_FN(linear_run));
    }
    */
}
