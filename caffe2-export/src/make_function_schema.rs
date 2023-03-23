crate::ix!();

#[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
pub const PREALLOCATED_OUTPUT_ARGNAME: &'static str = "_caffe2_preallocated_outputs";

#[cfg(any(expose_c2_ops, all(not(caffe2_is_xplat_build), not(c10_mobile))))]
#[inline] pub fn make_function_schema_for_c10(schema_str: *const u8) -> FunctionSchema {
    
    todo!();
    /*
        #if !defined(EXPOSE_C2_OPS) && \
        (defined(CAFFE2_IS_XPLAT_BUILD) || defined(C10_MOBILE))
      throw std::logic_error(
          "We don't support registering c10 ops on mobile yet because the function schema parser isn't present in the mobile build.");
    #else
      c10::FunctionSchema parsed_schema = torch::jit::parseSchema(schema_str);
      std::vector<c10::Argument> arguments = parsed_schema.arguments();
      arguments.emplace_back(
          PREALLOCATED_OUTPUT_ARGNAME,
          c10::OptionalType::create(c10::ListType::ofTensors()),
          nullopt,
          IValue());

      return FunctionSchema(
          parsed_schema.name(),
          parsed_schema.overload_name(),
          std::move(arguments),
          parsed_schema.returns(),
          parsed_schema.is_vararg(),
          parsed_schema.is_varret());
    #endif
    */
}
