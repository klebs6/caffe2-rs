crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/metal/MetalPrepackOpRegister.cpp]

pub fn unpack_a(
        weight:     Tensor,
        bias:       Option<Tensor>,
        stride:     Vec<i64>,
        padding:    Vec<i64>,
        dilation:   Vec<i64>,
        groups:     i64,
        output_min: &Option<Scalar>,
        output_max: &Option<Scalar>) -> IntrusivePtr<Conv2dOpContext> {
    
    todo!();
        /*
            auto packedWeight = weight.contiguous(MemoryFormat::ChannelsLast);
      return make_intrusive<Conv2dOpContext>(
          move(packedWeight),
          move(bias),
          stride,
          padding,
          dilation,
          groups,
          output_min,
          output_max);
        */
}

pub fn unpack_b(
        weight:     Tensor,
        bias:       Option<Tensor>,
        output_min: &Option<Scalar>,
        output_max: &Option<Scalar>) -> IntrusivePtr<LinearOpContext> {
    
    todo!();
        /*
            TORCH_CHECK(weight.dim() == 2);
      // Don't need to do `weight.t()`
      auto packedWeight = weight.view({weight.size(0), weight.size(1), 1, 1})
                              .contiguous(MemoryFormat::ChannelsLast);
      return make_intrusive<LinearOpContext>(
          move(packedWeight), move(bias), output_min, output_max);
        */
}

lazy_static!{
    /*
    TORCH_LIBRARY(metal, m) {
      m.class_<Conv2dOpContext>("Conv2dOpContext")
          .def_pickle(
              [](const intrusive_ptr<Conv2dOpContext>& op_context)
                  -> SerializationTypeConv2dPrePack { // __getstate__
                return op_context->pack();
              },
              [](SerializationTypeConv2dPrePack state)
                  -> intrusive_ptr<Conv2dOpContext> { // __setstate__
                return unpack(
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
              [](const intrusive_ptr<LinearOpContext>& op_context)
                  -> SerializationTypeLinearPrePack { // __getstate__
                return op_context->pack();
              },
              [](SerializationTypeLinearPrePack state)
                  -> intrusive_ptr<LinearOpContext> { // __setstate__
                return unpack(
                    move(get<0>(state)),
                    move(get<1>(state)),
                    get<2>(state),
                    get<3>(state));
              });
      m.def("copy_to_host(Tensor X) -> Tensor Y");
    }
    */
}

lazy_static!{
    /*
    TORCH_LIBRARY(metal_prepack, m) {
      m.def(
          "conv2d_prepack(Tensor W, Tensor? B, int[2] stride, "
          "int[2] padding, int[2] dilation, int groups, "
          "Scalar? output_min=None, Scalar? output_max=None) "
          "-> __torch__.torch.classes.metal.Conv2dOpContext");
      m.def(
          "conv2d_run(Tensor X, "
          "__torch__.torch.classes.metal.Conv2dOpContext W_prepack) -> Tensor Y");

      m.def(
          "linear_prepack(Tensor W, Tensor? B, Scalar? output_min=None, Scalar? output_max=None) -> __torch__.torch.classes.metal.LinearOpContext");

      m.def(
          "linear_run(Tensor X, __torch__.torch.classes.metal.LinearOpContext W_prepack) -> Tensor Y");
    }
    */
}

pub fn conv2d_prepack(
        weight:     Tensor,
        bias:       Option<Tensor>,
        stride:     Vec<i64>,
        padding:    Vec<i64>,
        dilation:   Vec<i64>,
        groups:     i64,
        output_min: &Option<Scalar>,
        output_max: &Option<Scalar>) -> IntrusivePtr<Conv2dOpContext> {
    
    todo!();
        /*
            TORCH_CHECK(weight.dim() == 4);
      return make_intrusive<Conv2dOpContext>(
          move(weight),
          move(bias),
          stride,
          padding,
          dilation,
          groups,
          output_min,
          output_max);
        */
}

pub fn linear_prepack(
        weight:     Tensor,
        bias:       Option<Tensor>,
        output_min: &Option<Scalar>,
        output_max: &Option<Scalar>) -> IntrusivePtr<LinearOpContext> {
    
    todo!();
        /*
            return make_intrusive<LinearOpContext>(
          move(weight), move(bias), output_min, output_max);
        */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(metal_prepack, CPU, m) {
      m.impl("conv2d_prepack", TORCH_FN(conv2d_prepack));
      m.impl("linear_prepack", TORCH_FN(linear_prepack));
    }
    */
}
