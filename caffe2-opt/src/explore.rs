crate::ix!();

/**
  | Explore the graph in topological order
  | until we hit stopping nodes. This is
  | based on Khan's algorithm:
  | 
  | https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
  | 
  | Precondition: nodes in `current_frontier`
  | must have satisfy `in_degree == 0`
  |
  */
#[inline] pub fn explore<T,U>(
    current_frontier: &Vec<NodeRef<T,U>>,
    context:          *mut VisitorContext<T,U>)  
{
    todo!();
    /*
        std::queue<NodeRef> q;
      for (const auto n : current_frontier) {
        q.push(n);
      }

      while (!q.empty()) {
        auto node = q.front();
        q.pop();
        auto& info = GetInfo(context->infos, node);

        // Check if the node is supported, stop exploring further if not supported
        if (nn::is<NeuralNetOperator>(node)) {
          const auto* nn_op = nn::get<NeuralNetOperator>(node);
          const auto& op_def =
              dyn_cast<Caffe2Annotation>(nn_op->getAnnotation())->getOperatorDef();
          bool wanted = context->predicate(op_def);
          wanted = context->find_supported ? wanted : (!wanted);
          if (!wanted) {
            context->frontier.emplace(node);
            continue;
          }
        }

        // Adding to current group
        info.group = context->group;
        info.needs_transform = context->find_supported;
        context->current_group.push_back(node);

        // Continue exploring its fanouts
        for (const auto& out_edge : node->getOutEdges()) {
          auto child_node = out_edge->head();
          auto& child_info = GetInfo(context->infos, child_node);
          if (--child_info.in_degree == 0) {
            q.push(child_node);
          }
        }
      }
    */
}

