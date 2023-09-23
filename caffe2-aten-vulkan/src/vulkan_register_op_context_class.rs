crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/vulkan/VulkanRegisterOpContextClass.cpp]

#[cfg(not(USE_VULKAN_API))]
lazy_static!{
    /*
    TORCH_LIBRARY(vulkan, m) {
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
    }
    */
}

#[cfg(not(USE_VULKAN_API))]
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
    }
    */
}

#[cfg(not(USE_VULKAN_API))]
lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(vulkan_prepack, CPU, m) {
      m.impl("conv2d_clamp_prepack", TORCH_FN(createConv2dClampPrePackOpContext));
    }
    */
}

#[cfg(not(USE_VULKAN_API))]
lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(vulkan_prepack, Vulkan, m) {
      m.impl("conv2d_clamp_run", convolution2d::conv2d_clamp_run);
    }
    */
}
