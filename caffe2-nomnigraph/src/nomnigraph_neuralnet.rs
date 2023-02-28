crate::ix!();

use crate::{
    NNMatchPredicate,
    NNCFGraph,
    NodeRef,
    NNSubgraph,
    NeuralNetOperator,
    NeuralNetData,
    NNModule
};

#[inline] pub fn has_producer<T,U>(n: NodeRef<T,U>) -> bool {
    
    todo!();
    /*
        return n->getInEdges().size() != 0;
    */
}

#[inline] pub fn get_producer<T,U>(n: NodeRef<T,U>) -> NodeRef<T,U> {
    
    todo!();
    /*
        assert(
          is<NeuralNetData>(n) &&
          "getProducer only works with NeuralNetData types.");
      auto inEdges = n->getInEdges();
      assert(inEdges.size() > 0 && "Tensor does not have a producer.");
      assert(
          inEdges.size() == 1 &&
          "Malformed NNGraph, NeuralNetData has multiple producers.");
      return inEdges.front()->tail();
    */
}

#[inline] pub fn has_consumer<T,U>(n: NodeRef<T,U>) -> bool {
    
    todo!();
    /*
        return n->getOutEdges().size() != 0;
    */
}

#[inline] pub fn get_consumers<T,U>(n: NodeRef<T,U>) -> Vec<NodeRef<T,U>> {
    
    todo!();
    /*
        assert(
          is<NeuralNetData>(n) &&
          "getProducer only works with NeuralNetData types.");
      std::vector<NodeRef> out;
      for (auto outEdge : n->getOutEdges()) {
        out.emplace_back(outEdge->head());
      }
      return out;
    */
}

#[inline] pub fn has_inputs<T,U>(n: NodeRef<T,U>) -> bool {
    
    todo!();
    /*
        return n->getInEdges().size() != 0;
    */
}

#[inline] pub fn get_inputs<T,U>(n: NodeRef<T,U>) -> Vec<NodeRef<T,U>> {
    
    todo!();
    /*
        assert(
          is<NeuralNetOperator>(n) &&
          "getInputs only works with NeuralNetOperator types.");
      std::vector<NodeRef> out;
      for (auto inEdge : n->getInEdges()) {
        out.emplace_back(inEdge->tail());
      }
      return out;
    */
}

#[inline] pub fn get_outputs<T,U>(n: NodeRef<T,U>) -> Vec<NodeRef<T,U>> {
    
    todo!();
    /*
        assert(
          is<NeuralNetOperator>(n) &&
          "getOutputs only works with NeuralNetOperator types.");
      std::vector<NodeRef> out;
      for (auto outEdge : n->getOutEdges()) {
        out.emplace_back(outEdge->head());
      }
      return out;
    */
}

#[inline] pub fn get_name<T,U>(n: NodeRef<T,U>) -> String {
    
    todo!();
    /*
        if (is<NeuralNetData>(n)) {
        return nn::get<NeuralNetData>(n)->getName();
      } else if (is<NeuralNetOperator>(n)) {
        return nn::get<NeuralNetOperator>(n)->getName();
      }
      return "Unknown";
    */
}

#[inline] pub fn get_subgraph_inputs<T,U>(subgraph: &NNSubgraph) -> HashSet<NodeRef<T,U>> {
    
    todo!();
    /*
        std::set<NodeRef> subgraph_inputs;
      for (const auto& node : subgraph.getNodes()) {
        NOM_REQUIRE_OR_CONT(is<NeuralNetData>(node));
        if (hasProducer(node)) {
          if (!subgraph.hasNode(getProducer(node))) {
            subgraph_inputs.insert(node);
          }
        } else {
          subgraph_inputs.insert(node);
        }
      }
      return subgraph_inputs;
    */
}

#[inline] pub fn get_subgraph_outputs<T,U>(subgraph: &NNSubgraph) -> HashSet<NodeRef<T,U>> {
    
    todo!();
    /*
        std::set<NodeRef> subgraph_outputs;
      for (const auto& n : subgraph.getNodes()) {
        NOM_REQUIRE_OR_CONT(is<NeuralNetData>(n));
        if (hasConsumer(n)) {
          for (const auto& consumer : getConsumers(n)) {
            if (!subgraph.hasNode(consumer)) {
              subgraph_outputs.insert(n);
            }
          }
        } else {
          subgraph_outputs.insert(n);
        }
      }
      return subgraph_outputs;
    */
}

#[inline] pub fn replace_producer<T,U>(
    tensor_node: NodeRef<T,U>, 
    new_producer: NodeRef<T,U>) 
{
    
    todo!();
    /*
        assert(
          is<NeuralNetData>(tensorNode) &&
          "First argument must contain NeuralNetData");
      auto inEdges = tensorNode->getInEdges();
      assert(
          inEdges.size() == 1 && "Tensor node passed in does not have a producer");
      auto edge = inEdges.at(0);
      auto prevProducer = edge->tail();
      prevProducer->removeOutEdge(edge);
      edge->setTail(newProducer);
      newProducer->addOutEdge(edge);
    */
}

#[inline] pub fn replace_all_uses_with<T,U>(
    old_tensor_node: NodeRef<T,U>, 
    new_tensor_node: NodeRef<T,U>)  {
    
    todo!();
    /*
        const auto edges = oldTensorNode->getOutEdges();
      for (const auto& edge : edges) {
        edge->setTail(newTensorNode);
        oldTensorNode->removeOutEdge(edge);
        newTensorNode->addOutEdge(edge);
      }
    */
}

#[inline] pub fn replace_as_consumer<T,U>(
    old_consumer: NodeRef<T,U>, 
    new_consumer: NodeRef<T,U>)
{
    todo!();
    /*
        const auto edges = oldConsumer->getInEdges();
      for (const auto& edge : edges) {
        edge->setHead(newConsumer);
        oldConsumer->removeInEdge(edge);
        newConsumer->addInEdge(edge);
      }
    */
}

#[inline] pub fn create_output<T,U>(
    nn:       *mut NNModule<T,U>,
    producer: NodeRef<T,U>,
    name:     String) -> NodeRef<T,U> {
    
    todo!();
    /*
        auto outputNode =
          nn->dataFlow.createNode(std::make_unique<Tensor>(name));
      nn->dataFlow.createEdge(producer, outputNode);
      return outputNode;
    */
}

/// Get all nodes tracked by CF graph
#[inline] pub fn get_tracked_nodes<T,U>(
    cf: &mut NNCFGraph) -> HashSet<NodeRef<T,U>> {
    
    todo!();
    /*
        std::unordered_set<NodeRef> cfTrackedNodes;
      for (const auto& bbNode : cf.getMutableNodes()) {
        auto& bb = bbNode->data();
        for (const auto node : bb.getInstructions()) {
          cfTrackedNodes.insert(node);
        }
      }
      return cfTrackedNodes;
    */
}

#[inline] pub fn coalesce_inserted_data_dependencies_helper<T,U>(m: *mut NNModule<T,U>) -> usize {
    
    todo!();
    /*
        auto cfTrackedNodes = getTrackedNodes(m->controlFlow);

      for (auto& bbNode : m->controlFlow.getMutableNodes()) {
        auto bb = bbNode->mutableData();
        // We mutate the instructions of the bb, so we copy here.
        // TODO make this an iterator and simply promote it on insertion.
        auto instrsCopy = bb->getInstructions();
        for (const auto instr : instrsCopy) {
          for (const auto input : repr::nn::getInputs(instr)) {
            if (!repr::nn::hasProducer(input)) {
              continue;
            }
            auto producer = repr::nn::getProducer(input);
            if (!cfTrackedNodes.count(producer)) {
              bb->insertInstructionBefore(producer, instr);
              cfTrackedNodes.insert(producer);
            }
          }
        }
      }

      return cfTrackedNodes.size();
    */
}

/// TODO: move this to more generic location.
/// TODO: [algo] improve this algorithm, as it is horrendously inefficient.
#[inline] pub fn coalesce_inserted_data_dependencies<T,U>(m: *mut NNModule<T,U>)  {
    
    todo!();
    /*
        size_t oldSize = 0;
      size_t newSize = 0;
      do {
        oldSize = newSize;
        newSize = coalesceInsertedDataDependenciesHelper(m);
      } while (newSize != oldSize);

      // Now we track new nodes that have no relationship to the old CFGraph
      auto cfTrackedNodes = getTrackedNodes(m->controlFlow);
      std::unordered_set<NodeRef> dfNodes;
      for (auto node : m->dataFlow.getMutableNodes()) {
        if (repr::nn::is<NeuralNetOperator>(node) && !cfTrackedNodes.count(node)) {
          dfNodes.insert(node);
        }
      }

      auto newBbNode = m->controlFlow.createAnonymousFunction();
      auto sccs = algorithm::tarjans(&m->dataFlow);
      for (auto iter = sccs.rbegin(); iter != sccs.rend(); ++iter) {
        for (auto node : iter->getNodes()) {
          if (dfNodes.count(node)) {
            auto currentBasicBlock = newBbNode->mutableData();
            currentBasicBlock->pushInstructionNode(node);
          }
        }
      }

      // Finally we reconcile any data dependency issues (if we can).
      for (auto& bbNode : m->controlFlow.getMutableNodes()) {
        auto bb = bbNode->mutableData();
        int permutation;
        do {
          permutation = 0;
          std::unordered_set<NodeRef> seen;
          for (auto instr_iter = bb->getMutableInstructions()->begin();
               instr_iter != bb->getMutableInstructions()->end();
               ++instr_iter) {
            // This cannot be auto& because *iter is pure R-ref
            auto instr = *instr_iter;
            for (auto& output : getOutputs(instr)) {
              for (auto& consumer : getConsumers(output)) {
                if (seen.count(consumer)) {
                  bb->moveInstructionBefore(instr, consumer);
                  ++permutation;
                }
              }
            }
            seen.insert(instr);
          }
        } while (permutation);
      }
    */
}

#[inline] pub fn has_single_output_and_consumer<T,U>(node_ref: NodeRef<T,U>) -> bool {
    
    todo!();
    /*
        auto nodeOutputs = nn::getOutputs(nodeRef);
      NOM_REQUIRE_OR_RET_FALSE(nodeOutputs.size() == 1);
      auto nodeConsumers = nn::getConsumers(nodeOutputs.front());
      return nodeConsumers.size() == 1;
    */
}

#[inline] pub fn has_unique_consumer<T,U>(node_ref: NodeRef<T,U>) -> bool {
    
    todo!();
    /*
        auto nodeOutputs = nn::getOutputs(nodeRef);
      NodeRef nodeConsumer = nullptr;
      for (auto nodeOutput : nodeOutputs) {
        for (auto consumer : nn::getConsumers(nodeOutput)) {
          if (nodeConsumer && consumer && consumer != nodeConsumer) {
            return false;
          }
          nodeConsumer = consumer;
        }
      }
      return true;
    */
}

#[inline] pub fn match_external_tensor_node() -> NNMatchPredicate {
    
    todo!();
    /*
        return NNMatchPredicate(nn::is<Tensor>).nonTerminal().excludeFromSubgraph();
    */
}
