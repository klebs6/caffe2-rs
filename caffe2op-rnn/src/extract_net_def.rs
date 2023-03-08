crate::ix!();

#[inline] pub fn extract_net_def(
    op:       &OperatorDef,
    arg_name: &String) -> NetDef 
{
    todo!();
    /*
        if (ArgumentHelper::HasSingleArgumentOfType<OperatorDef, NetDef>(
              op, argName)) {
        return ArgumentHelper::GetSingleArgument<OperatorDef, NetDef>(
            op, argName, NetDef());
      } else {
    #ifndef CAFFE2_RNN_NO_TEXT_FORMAT
        NetDef result;
        const auto netString =
            ArgumentHelper::GetSingleArgument<OperatorDef, string>(op, argName, "");
        CAFFE_ENFORCE(
            TextFormat::ParseFromString(netString, &result),
            "Invalid NetDef");
        return result;
    #else
        CAFFE_THROW("No valid NetDef for argument ", argName);
    #endif
      }
    */
}
