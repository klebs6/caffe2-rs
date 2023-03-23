crate::ix!();

pub type NNGraph    = NomGraph<Box<Value>>;
pub type NNSubgraph = Subgraph<Box<Value>>;
pub type NNCFGraph  = ControlFlowGraph<NNGraph>;

pub struct NNModule<T,U> {
    data_flow:     NNGraph,
    control_flow:  NNCFGraph,
    inputs:        HashSet<NodeRef<T,U>>,
    outputs:       HashSet<NodeRef<T,U>>,
}

impl<T,U> NNModule<T,U> {
    
    /**
      | Simple wrapper of replaceSubgraph where
      | the node is created for you.
      |
      | Returns a NodeRef to the node containing
      | the operator that was created
      */
    #[inline] pub fn replace_subgraph_with_operator<Args>(&mut self, 
        sg:               &SubgraphType<T,U>,
        subgraph_inputs:  &Vec<NodeRef<T,U>>,
        subgraph_outputs: &Vec<NodeRef<T,U>>,
        args:             Args) -> NodeRef<T,U> {

        todo!();
        /*
            auto node = dataFlow.createNode(std::make_unique<T>(args));
      replaceSubgraph(sg, node, subgraph_inputs, subgraph_outputs);
      return node;
        */
    }
    
    #[inline] pub fn create_unique_data_node(
        &mut self, 
        s: Option<&str>) -> NodeRef<T,U> {
        
        let s = s.unwrap_or("_unique");

        todo!();
        /*
            auto curr_name = s;
      auto iter = 0;
      bool need_name = true;
      do {
        need_name = false;
        for (const auto& node : dataFlow.getMutableNodes()) {
          if (nn::getName(node) == curr_name) {
            std::stringstream ss;
            ss << iter;
            curr_name = s + "_" + ss.str();
            iter++;
            need_name = true;
            break;
          }
        }
      } while (need_name);
      return dataFlow.createNode(std::make_unique<Tensor>(curr_name));
        */
    }
    
    /**
      | Replace subgraph sg by node, using the
      | order of node_inputs and node_outputs
      | to determine how to link them to the node.
      | node_inputs *must* enumerate all the
      | inputs to the subgraph (NeuralNetData
      | that do not have producers inside the
      | subgraph). Same for node_outputs
      | 
      | New output names may be created in the
      | case that an inputs and an output have
      | the same name (to avoid in place ops).
      | 
      | This may cause issues with external_output
      | 
      | -- be sure to check after running this
      | function (and perhaps inserting a copy/alias
      | op).
      |
      */
    #[inline] pub fn replace_subgraph(&mut self, 
        subgraph:     &NNSubgraph,
        node:         &NodeRef<T,U>,
        node_inputs:  &Vec<NodeRef<T,U>>,
        node_outputs: &Vec<NodeRef<T,U>>)  {

        todo!();
        /*
            auto sg = subgraph;
      auto sg_inputs = nn::getInputs(sg);
      auto sg_outputs = nn::getOutputs(sg);

      auto sg_inputs_copy = sg_inputs;
      auto sg_outputs_copy = sg_outputs;

      for (const auto& input : node_inputs) {
        sg_inputs_copy.erase(input);
        // outputs may contain inputs that have additional
        // consumers external to the subgraph
        sg_outputs_copy.erase(input);
      }
      assert(sg_inputs_copy.size() == 0 && "Not all inputs were listed");

      for (const auto& output : node_outputs) {
        sg_outputs_copy.erase(output);
      }
      assert(sg_outputs_copy.size() == 0 && "Not all outputs were listed");

      for (auto& input : node_inputs) {
        dataFlow.createEdge(input, node);
        sg.removeNode(input);
      }
      for (auto& output : node_outputs) {
        if (sg_inputs.count(output)) {
          dataFlow.createEdge(node, createUniqueDataNode());
          continue;
        }
        dataFlow.createEdge(node, output);
        sg.removeNode(output);
      }
      deleteSubgraph(sg);
        */
    }
    
    #[inline] pub fn delete_subgraph(&mut self, subgraph: &NNSubgraph)  {
        
        todo!();
        /*
            dataFlow.deleteNodes(subgraph.getNodes());
        */
    }
}
