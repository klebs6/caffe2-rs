#![feature(specialization)]

#![feature(test)]
extern crate test;

#[macro_use] mod imports;
use imports::*;

x!{nomnigraph_algorithmstest}
x!{nomnigraph_binarymatchimpltest}
x!{nomnigraph_converters_dot}
x!{nomnigraph_generated_opclasses}
x!{nomnigraph_graph_algorithms}
x!{nomnigraph_graph_binarymatchimpl}
x!{nomnigraph_graph_graph}
x!{nomnigraph_graph_tarjansimpl}
x!{nomnigraph_graph_toposort}
x!{nomnigraph_graphtest}
x!{nomnigraph_matchtest}
x!{nomnigraph_neuralnettest}
x!{nomnigraph_neuralnet}
x!{nomnigraph_representations_compiler}
x!{nomnigraph_representations_controlflow}
x!{nomnigraph_representations_neuralnet}
x!{nomnigraph_subgraphmatchertest}
x!{nomnigraph_support_common}
x!{nomnigraph_tarjansimpltest}
x!{nomnigraph_test_util}
x!{nomnigraph_toposorttest}
x!{nomnigraph_transformations_match}
x!{nomnigraph_transformations_subgraphmatcher}
