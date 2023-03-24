crate::ix!();

#[inline] pub fn chain_error_net(
    ws:       *mut Workspace,
    net_name: &String,
    throw:    bool) -> Box<NetBase> {
    
    todo!();
    /*
        std::string spec_template = R"DOC(
            name: "<NET_NAME>"
            type: "async_scheduling"
            op {
              type: "SyncErrorOp"
              arg {
                name: "fail"
                i: 1
              }
              arg {
                name: "throw"
                i: <THROW>
              }
            }
            op {
              type: "SyncErrorOp"
              arg {
                name: "fail"
                i: 0
              }
            }
      )DOC";

      std::string spec = spec_template;
      ReplaceAll(spec, "<NET_NAME>", net_name.c_str());
      ReplaceAll(spec, "<THROW>", throw_ ? "1" : "0");

      NetDef net_def;
      CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));
      return CreateNet(net_def, ws);
    */
}


