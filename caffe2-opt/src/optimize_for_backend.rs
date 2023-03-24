crate::ix!();

pub type SupportsFn  = fn(opdef: &OperatorDef) -> bool;
pub type TransformFn = fn(netdef: &NetDef)     -> NetDef;

pub fn optimize_for_backend(
    net:            &mut NetDef, 
    supports:       SupportsFn, 
    transform_func: TransformFn, 
    debug:          bool) -> NetDef 
{
    todo!();

    /*
      auto nn = convertToNNModule(net);
      auto& dfg = nn.dataFlow;

      // Initialize the group info and figure out the external/input output
      VisitorContext context(supports);
      std::vector<NodeRef> external_inputs;
      std::unordered_set<std::string> external_outputs;
      for (auto node : dfg.getMutableNodes()) {
        context.infos.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(node),
            std::forward_as_tuple(node->getInEdges().size(), -1));

        if (!nn::is<NeuralNetOperator>(node)) {
          if (!nn::hasProducer(node)) {
            external_inputs.push_back(node);
          }
          if (!nn::hasConsumer(node)) {
            external_outputs.emplace(nn::get<const NeuralNetData>(node)->getName());
          }
          for (auto i = 0; i < net.external_output_size(); ++i) {
            const auto& n = net.external_output(i);
            if (n == nn::get<const NeuralNetData>(node)->getName()) {
              external_outputs.emplace(n);
            }
          }
        }
      }

      // Find unsupported and supported groups of nodes alternatively
      context.frontier.clear();
      context.current_group.clear();
      context.find_supported = false;
      std::vector<TransformSubgraph> subs;
      for (std::vector<NodeRef> frontier(
               external_inputs.begin(), external_inputs.end());
           !frontier.empty();
           context.find_supported = !context.find_supported) {
        Explore(frontier, &context);
        if (context.find_supported) {
          subs.emplace_back(
              std::move(frontier),
              std::move(context.current_group),
              context.group,
              context.find_supported);
        }

        frontier.assign(context.frontier.begin(), context.frontier.end());
        context.frontier.clear();
        context.current_group.clear();
        context.group++;
      }

      // Transform needed subgraphs one by one
      std::vector<caffe2::NetDef> opt_subnets;
      opt_subnets.reserve(subs.size());
      for (auto& g : subs) {
        // Generate boundary input/output edges
        DetectBoundaryReferences(&g, context.infos, external_outputs);

        caffe2::NetDef subnet = ConvertToC2Net(g, context.infos);
        // Transform the subgraph protobuf def, note that we can have less external
        // inputs/outputs but not more
        opt_subnets.emplace_back(transform_func(subnet));

        ReplaceSubgraph(g, opt_subnets.back(), &dfg);
      }

      // Prune dangling nodes, because after transformation, some weights might be
      // absorbed
      PruneUnrefereredNodes(&nn);

      if (debug) {
        DumpGraph(&dfg, "dump.dot");
      }

      auto new_net = convertToCaffe2Proto(nn);
      new_net.set_name(net.name() + "_opt");
      return new_net;

    */
}
