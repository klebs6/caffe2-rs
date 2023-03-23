crate::ix!();

/// op schema check
#[inline] pub fn run_schema_check(net: &NetDef)  {
    
    todo!();
    /*
        for (auto& op : net.op()) {
        auto* schema = OpSchemaRegistry::Schema(op.type());
        if (schema) {
          CAFFE_ENFORCE(
              schema->Verify(op),
              "Operator def did not pass schema checking: ",
              ProtoDebugString(op));
        }
      }
    */
}
