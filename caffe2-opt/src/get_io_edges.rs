crate::ix!();

#[inline] pub fn get_input_edges(
    sg: &NNGraph_SubgraphType, 
    g:  &NNGraph) -> Vec<NNGraph_EdgeRef> 
{
    todo!();
    /*
        std::vector<NNGraph_EdgeRef> inputTensorEdges;
      for (const auto& node : sg.getNodes()) {
        NOM_REQUIRE_OR_CONT(nn::is<NeuralNetOperator>(node));
        NOM_REQUIRE_OR_CONT(nn::hasInputs(node));

        // Check if tensor's parents are in the sg
        for (const auto& input : nn::getInputs(node)) {
          NOM_REQUIRE_OR_CONT(
              !nn::hasProducer(input) || !sg.hasNode(nn::getProducer(input)));
          inputTensorEdges.emplace_back(g.getEdge(input, node));
        }
      }
      return inputTensorEdges;
    */
}

#[inline] pub fn get_output_edges(
    sg: &NNGraph_SubgraphType, 
    g: &NNGraph) -> Vec<NNGraph_EdgeRef> 
{
    todo!();
    /*
        std::vector<NNGraph_EdgeRef> outputTensorEdges;
      for (const auto& node : sg.getNodes()) {
        NOM_REQUIRE_OR_CONT(nn::is<NeuralNetOperator>(node));

        for (const auto& output : nn::getOutputs(node)) {
          auto consumers = nn::getConsumers(output);
          for (const auto& consumer : consumers) {
            NOM_REQUIRE_OR_CONT(!sg.hasNode(consumer));
            outputTensorEdges.emplace_back(g.getEdge(node, output));
          }
          NOM_REQUIRE_OR_CONT(consumers.size() == 0);
          outputTensorEdges.emplace_back(g.getEdge(node, output));
        }
      }
      return outputTensorEdges;
    */
}
