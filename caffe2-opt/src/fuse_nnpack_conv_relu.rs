crate::ix!();

#[inline] pub fn fuse_nnpackconv_relu<T,U>(nn: *mut NNModule<T,U>)  {
    
    todo!();
    /*
        auto should_fuse = [](const repr::Conv& conv) {
        const auto annotation = conv.getAnnotation();
        if (!annotation || !isa<Caffe2Annotation>(annotation)) {
          return false;
        }
        const auto& op = dyn_cast<Caffe2Annotation>(annotation)->getOperatorDef();

        // We only want to fuse for fast NNPACK convs
        if (op.engine() != "NNPACK") {
          return false;
        }
        caffe2::string algo = "AUTO";
        for (const auto &arg : op.arg()) {
          if (arg.name() == "algo") {
            algo = arg.s();
          }
        }
        if (!isNNPACKConvReluEfficient(algo, conv)) {
          return false;
        }
        return true;
      };

      auto postprocess = [](repr::NNGraph::NodeRef conv_node) {
        auto conv = repr::nn::get<repr::Conv>(conv_node);
        auto annotation = conv->getMutableAnnotation();
        if (!annotation || !isa<Caffe2Annotation>(annotation)) {
          return;
        }
        auto* op = dyn_cast<Caffe2Annotation>(annotation)->getMutableOperatorDef();
        auto* arg = op->add_arg();
        arg->set_name("activation");
        arg->set_s("Relu");
      };

      fuseActivation<repr::Conv, repr::Relu>(nn, should_fuse, postprocess);
    */
}


