crate::ix!();

#[inline] pub fn create(net: *const NetBase, net_name: &String) -> Arc<Tracer> {
    
    todo!();
    /*
        // Enable the tracer if the net has the "enable_tracing" argument set OR
      // if the command line option includes the net name option in the list of
      // traceable nets.
      bool trace_net = hasEnableTracingFlag(net) || isTraceableNetName(net_name);
      return trace_net
          ? std::make_shared<Tracer>(net, net_name, getTracingConfigFromNet(net))
          : nullptr;
    */
}
