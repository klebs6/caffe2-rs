crate::ix!();

#[inline] pub fn addNNPACK<T,U>(nn: *mut NNModule<T,U>, low_memory: Option<bool>)  {

    let low_memory = low_memory.unwrap_or(false);
    
    todo!();
    /*
        for (auto node : nn->dataFlow.getMutableNodes()) {
        // Skip blobs.
        NOM_REQUIRE_OR_CONT(repr::nn::is<repr::NeuralNetOperator>(node));

        // Check if it is a convolution.
        auto nnOp = repr::nn::get<repr::NeuralNetOperator>(node);
        NOM_REQUIRE_OR_CONT(isa<nom::repr::Conv>(nnOp));

        // Requires X, W, b for NNPACK
        NOM_REQUIRE_OR_CONT(node->getInEdges().size() >= 3);

        std::string engine = "NNPACK";

        // Now do some specific checks to see if an NNPACK engine is correct.
        bool validTransformCandidate = true;
        auto conv = dyn_cast<nom::repr::Conv>(nnOp);

        NOM_REQUIRE_OR_CONT(conv->getLayout() == nom::repr::Conv::NNLayout::NCHW);

        // NNPACK only supports stride == 1
        for (auto stride : conv->getStrides()) {
          if (stride != 1) {
            validTransformCandidate = false;
            break;
          }
        }
        NOM_REQUIRE_OR_CONT(validTransformCandidate);

        // NNPACK only supports 2DConv.
        const auto& kernelShape = conv->getKernelShape();
        NOM_REQUIRE_OR_CONT(kernelShape.size() == 2);

        // Kx1 and 1xK convs are inefficient in NNPACK.
        if (kernelShape[0] != kernelShape[1]) {
          NOM_REQUIRE_OR_CONT(kernelShape[0] != 1 && kernelShape[1] != 1);
        }

        // We're good to use our engine.
        auto annotation = conv->getMutableAnnotation();
        NOM_REQUIRE_OR_CONT(annotation && isa<Caffe2Annotation>(annotation));

        auto* op = dyn_cast<Caffe2Annotation>(annotation)->getMutableOperatorDef();
        op->set_engine(engine);
        if (!low_memory) {
          auto* precompute_argument = op->add_arg();
          precompute_argument->set_name("convolution_transform_strategy");
          precompute_argument->set_s("PRECOMPUTE");
        }
      }
    */
}


