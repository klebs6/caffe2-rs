crate::ix!();

/// Convert a graph to dot string.
#[inline] pub fn convert_graph_to_dot_string<G: GraphType>(
    g:            *mut G,
    node_printer: NodePrinter<G>,
    edge_printer: Option<EdgePrinter<G>>) -> String {


    todo!();
    /*
    let edge_printer: EdgePrinter<G> =
             edge_printer.unwrap_or(DotGenerator<G>::default_edge_printer);

        auto d = DotGenerator<GraphT>(nodePrinter, edgePrinter);
      return d.convert(algorithm::createSubgraph(g), {});
    */
}

/**
  | Convert a graph to dot string and annotate
  | subgraph clusters.
  |
  */
#[inline] pub fn convert_graph_to_dot_string_and_annotate_subgraph_clusters<G: GraphType>(
    g:            *mut G,
    subgraphs:    &Vec<*mut <G as GraphType>::SubgraphType>,
    node_printer: NodePrinter<G>,
    edge_printer: EdgePrinter<G>) -> String {

    todo!();
    /*

    let edge_printer: EdgePrinter<G> =
             edge_printer.unwrap_or(DotGenerator<G>::default_edge_printer);

        auto d = DotGenerator<GraphT>(nodePrinter, edgePrinter);
      return d.convert(algorithm::createSubgraph(g), subgraphs);
    */
}

/// Convert a subgraph to dot string.
#[inline] pub fn convert_subgraph_to_dot_string<G: GraphType>(
    sg:           &<G as GraphType>::SubgraphType,
    node_printer: NodePrinter<G>,
    edge_printer: EdgePrinter<G>) -> String {


    todo!();
    /*
        let edge_printer: DotGenerator<G>::EdgePrinter =
                 edge_printer.unwrap_or(DotGenerator::<G>::default_edge_printer);

        auto d = DotGenerator<GraphT>(nodePrinter, edgePrinter);
      return d.convert(sg);
    */
}

#[inline] pub fn convert_to_dot_record_string<G: GraphType>(
    g:            *mut G,
    node_printer: NodePrinter<G>,
    edge_printer: EdgePrinter<G>) -> String {

    todo!();
    /*
        let edge_printer: DotGenerator<G>::EdgePrinter =
                 edge_printer.unwrap_or(DotGenerator::<G>::default_edge_printer);

        auto d = DotGenerator<GraphT>(nodePrinter, edgePrinter);
      return d.convertStruct(algorithm::createSubgraph(g));
    */
}
