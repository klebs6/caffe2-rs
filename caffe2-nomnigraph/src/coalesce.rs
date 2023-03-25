crate::ix!();

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
