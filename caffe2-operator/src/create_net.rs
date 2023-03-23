crate::ix!();

pub const kSimpleNet: &'static str = "simple";

/**
  | -----------
  | @brief
  | 
  | Creates a network, accessing / creating
  | blobs in the given workspace.
  | 
  | -----------
  | @note
  | 
  | this is different from Workspace::CreateNet.
  | The latter adds the created net object
  | to the workspace's net map, while this
  | function returns a standalone net object.
  |
  */
#[inline] pub fn create_net(
    net_def: &Arc<NetDef>,
    ws:      *mut Workspace) -> Box<NetBase> 
{
    todo!();
    /*
        std::string net_type;
      if (net_def->has_type() && !net_def->type().empty()) {
        net_type = net_def->type();
      } else {
        // By default, we will return a simple network that just runs all operators
        // sequentially.
        net_type = kSimpleNet;
      }
      ApplyPotentialExecutorOverride(&net_type);
      unique_ptr<NetBase> net = NetRegistry()->Create(net_type, net_def, ws);

      VLOG(1) << "Adding a global observer to a net";
      if (net) {
        auto* observer_creators = GetNetObserverCreators();
        for (auto& creator : *observer_creators) {
          net->AttachObserver(creator(net.get()));
        }
      }
      return net;
    */
}

