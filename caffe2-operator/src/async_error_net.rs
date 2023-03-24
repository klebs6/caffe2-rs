crate::ix!();

#[inline] pub fn async_error_net(
    ws:           *mut Workspace,
    net_name:     &String,
    throw:        bool,
    fail_in_sync: bool) -> Box<NetBase> {
    
    todo!();
    /*
        std::string spec_template = R"DOC(
            name: "<NET_NAME>"
            type: "async_scheduling"
            op {
              type: "AsyncErrorOp"
              arg {
                name: "throw"
                i: <THROW>
              }
              arg {
                name: "fail_in_sync"
                i: <FAIL_IN_SYNC>
              }
            }
      )DOC";

      std::string spec = spec_template;
      ReplaceAll(spec, "<NET_NAME>", net_name.c_str());
      ReplaceAll(spec, "<THROW>", throw_ ? "1" : "0");
      ReplaceAll(spec, "<FAIL_IN_SYNC>", fail_in_sync ? "1" : "0");

      NetDef net_def;
      CAFFE_ENFORCE(TextFormat::ParseFromString(spec, &net_def));
      return CreateNet(net_def, ws);
    */
}
