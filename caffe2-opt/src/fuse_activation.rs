crate::ix!();

/**
  | Generic activation fusion helper.
  | 
  | -----------
  | @param OperationT
  | 
  | The operator to be fused.
  | ----------
  | @param ActivationT
  | 
  | The activation to be fused.
  | ----------
  | @param nn
  | 
  | Neural network module to be modified
  | in place
  | ----------
  | @param should_fuse
  | 
  | Given a conv op, check whether we want
  | to fuse it with subsequent relu or not
  | ----------
  | @param postprocess
  | 
  | Functor to postprocess the conv node,
  | attaching additional attributes if
  | necessary
  |
  */
pub fn fuse_activation<OperationT, ActivationT,T,U>(
    nn:          *mut NNModule<T,U>,
    should_fuse: fn(conv: &OperationT) -> bool,
    postprocess: fn(conv_node: NodeRef<T,U>) -> ()) {
    todo!();
    /*
    for (auto node_pair : repr::nn::dataIterator<OperationT>(nn->dataFlow)) {
        repr::NNGraph::NodeRef conv_node;
        OperationT* conv;
        std::tie(conv, conv_node) = node_pair;

        // Check topological feasibility
        auto conv_outputs = repr::nn::getOutputs(conv_node);
        if (conv_outputs.size() != 1) {
            continue;
        }
        auto conv_output = conv_outputs.front();

        auto consumers = repr::nn::getConsumers(conv_output);
        if (consumers.size() != 1) {
            continue;
        }
        if (!repr::nn::is<ActivationT>(consumers.front())) {
            continue;
        }
        auto relu_node = consumers.front();

        auto relu_outputs = repr::nn::getOutputs(relu_node);
        if (relu_outputs.size() != 1) {
            continue;
        }

        // Check feasibility with application specific logic
        if (!should_fuse(*conv)) {
            continue;
        }

        // Ready to fuse
        auto relu_output = relu_outputs.front();
        auto output_tensor = repr::nn::get<repr::Tensor>(relu_output);
        auto output_node = relu_output;
        auto input_tensor =
            repr::nn::get<repr::Tensor>(repr::nn::getInputs(conv_node).front());

        // Conv cannot be in-place
        if (output_tensor->getName() != input_tensor->getName()) {
            nn->dataFlow.replaceNode(conv_output, relu_output);
            nn->dataFlow.deleteNode(relu_node);
            nn->dataFlow.deleteNode(conv_output);
        } else {
            nn->dataFlow.replaceNode(relu_output, conv_output);
            output_tensor = repr::nn::get<repr::Tensor>(conv_output);
            output_node = conv_output;
            nn->dataFlow.deleteNode(relu_node);
            nn->dataFlow.deleteNode(relu_output);
        }

        // We may have accidentally made the next op in-place
        // In future iterations of transformations this won't be an issue,
        // but current caffe2 predictor usage requires things like
        // external_input and output to be unchanged.
        bool rectify_inplace = false;
        for (auto& consumer : repr::nn::getConsumers(output_node)) {
            for (auto& consumer_output : repr::nn::getOutputs(consumer)) {
                auto co_name = repr::nn::get<repr::Tensor>(consumer_output)->getName();
                if (co_name == output_tensor->getName()) {
                    rectify_inplace = true;
                }
            }
        }
        if (rectify_inplace) {
            auto new_output = nn->dataFlow.createNode(
                make_unique<repr::Tensor>(output_tensor->getName() + "_fusion_fix"));
            nn->dataFlow.replaceNode(output_node, new_output);
        }

        // Application specific logic for postprocessing the conv node
        postprocess(conv_node);
    }
    */
}

/// $$ X_{bn} = \frac{s(X - m)}{\sqrt{\sigma + \epsilon}} + b_{bn}$$
/// $$ X_{conv} = X * W + b_{conv} $$
/// thus, substituting $X$ with $X_{conv}$ in the BN equation we get:
/// $$X_{bn} = X * \frac{sW}{\sqrt{\sigma + \epsilon}} + \frac{s(b_{conv} -
/// m)}{\sqrt{\sigma + \epsilon}} + b_{bn}$$ or
/// $$ W' = W\frac{s}{\sqrt{\sigma + \epsilon}}$$
/// $$ b' = (b_{conv} - m)\frac{s}{\sqrt{\sigma + \epsilon}} + b_{bn}$$
#[inline] pub fn fuse_conv_bnhelper<T,U>(nn: *mut NNModule<T,U>, ws: *mut Workspace) -> bool {
    
    todo!();
    /*
        size_t convOrder = 0;
      for (auto node_pair : repr::nn::dataIterator<repr::Conv>(nn->dataFlow)) {
        repr::NNGraph::NodeRef convNode;
        repr::Conv* conv;
        std::tie(conv, convNode) = node_pair;

        auto output = repr::nn::getOutputs(convNode).front();
        auto consumers = repr::nn::getConsumers(output);
        NOM_REQUIRE_OR_CONT(consumers.size() == 1);

        auto consumer = consumers.front();
        NOM_REQUIRE_OR_CONT(repr::nn::is<repr::BatchNormalization>(consumer));

        auto bnNode = consumer;
        auto bn = repr::nn::get<repr::BatchNormalization>(bnNode);
        auto bnOutputs = nn::getOutputs(bnNode);
        NOM_REQUIRE_OR_CONT(bnOutputs.size() == 1);
        auto bnOutput = bnOutputs.front();

        auto convInputs = repr::nn::getInputs(convNode);
        if (convInputs.size() < 2) {
          continue;
        }

        auto bnInputs = repr::nn::getInputs(bnNode);
        CAFFE_ENFORCE(
            bnInputs.size() >= 5, "Invalid batch normalization input size");

    #define EXPOSE_TENSOR_DATA(name, index, inputs)                                \
      auto name = repr::nn::get<repr::Tensor>(inputs[index]);                      \
      assert(ws->HasBlob(name->getName()) && "Blob not in workspace");             \
      auto name##Tensor = BlobGetMutableTensor(ws->GetBlob(name->getName()), CPU); \
      auto name##Data = name##Tensor->mutable_data<float>();

        EXPOSE_TENSOR_DATA(filter, 1, convInputs);

        EXPOSE_TENSOR_DATA(scale, 1, bnInputs);
        EXPOSE_TENSOR_DATA(biasBN, 2, bnInputs);
        EXPOSE_TENSOR_DATA(mean, 3, bnInputs);
        EXPOSE_TENSOR_DATA(variance, 4, bnInputs);

        if (convInputs.size() == 2) {
          NOM_REQUIRE_OR_CONT(conv->getMutableAnnotation() != nullptr);
          auto annotation =
              dyn_cast<caffe2::Caffe2Annotation>(conv->getMutableAnnotation());
          NOM_REQUIRE_OR_CONT(annotation != nullptr);
          auto op = annotation->getOperatorDef();
          auto convName = op.name();

          while (true) {
            auto convBiasName = convName + "_bias" + to_string(convOrder);
            if (!ws->HasBlob(convBiasName)) {
              auto convBiasTensor = make_unique<repr::Tensor>(convBiasName);
              convBiasTensor->setType(repr::Tensor::DataType::Float);
              auto convBiasNode = nn->dataFlow.createNode(
                  unique_dyn_cast<repr::NeuralNetData>(convBiasTensor));
              nn->inputs.insert(convBiasNode);
              nn->dataFlow.createEdge(convBiasNode, convNode);

              auto* blob = ws->CreateBlob(convBiasName);
              caffe2::TensorCPU* tensor = BlobGetMutableTensor(blob, caffe2::CPU);
              CHECK_NOTNULL(tensor);
              // Get output channel
              size_t c = filterTensor->dim32(0);
              tensor->Resize(c);
              float* tensor_data = tensor->mutable_data<float>();
              memset(tensor_data, 0, tensor->nbytes());
              break;
            }
            convOrder++;
          }
        }

        convInputs = repr::nn::getInputs(convNode);
        EXPOSE_TENSOR_DATA(biasConv, 2, convInputs);

    #undef EXPOSE_TENSOR_DATA

        // Assume M{CHW,HWC}
        auto chwDim = filterTensor->size_from_dim(1);
        for (auto c = 0; c < filterTensor->dim32(0); ++c) {
          float coeff =
              scaleData[c] / std::sqrt(varianceData[c] + bn->getEpsilon());
          for (auto i = 0; i < chwDim; ++i) {
            filterData[c * chwDim + i] *= coeff;
          }
          auto bias = (biasConvData[c] - meanData[c]) * coeff + biasBNData[c];
          biasConvData[c] = bias;
        }

        nn->dataFlow.deleteNode(output);
        nn->dataFlow.createEdge(convNode, bnOutput);
        nn->dataFlow.deleteNode(bnNode);
        return true;
      }
      return false;
    */
}

#[inline] pub fn fuse_convBN<T,U>(nn: *mut NNModule<T,U>, ws: *mut Workspace)  {
    
    todo!();
    /*
        while (fuseConvBNHelper(nn, ws)) {
      }
    */
}

register_ws_opt_pass_from_func!{FuseConvBN, fuseConvBN}

