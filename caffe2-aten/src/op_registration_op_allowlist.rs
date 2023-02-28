/*!
 | This header implements functionality to build
 | PyTorch with only a certain set of operators (+
 | dependencies) included.
 |
 | - Build with
 |   -DTORCH_OPERATOR_WHITELIST="add;sub" and only
 |   these two ops will be included in your build.
 |   The allowlist records operators only, no
 |   overloads; if you include add, all overloads
 |   of add will be included.
 |
 | Internally, this is done by removing the
 | operator registration calls using compile time
 | programming, and the linker will then prune all
 | operator functions that weren't registered. See
 | Note [Selective build] for more details
 |
 | WARNING: The allowlist mechanism doesn't work
 | for all ways you could go about registering an
 | operator.  If the dispatch key / operator name
 | is not sufficiently obvious at compile time,
 | then the allowlisting mechanism will fail (and
 | the operator will be included in the binary
 | anyway).
 */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/op_registration/op_allowlist.h]

/**
  | returns true iff allowlist contains item
  |
  | op_allowlist_contains("a;bc;d", "bc") == true
  |
  */
pub fn op_allowlist_contains(
        allowlist: StringView,
        item:      StringView) -> bool {
    
    todo!();
        /*
            //Choose a really big value for next so that if something goes wrong
        //this code will blow up in a hopefully detectable way.
        usize next = usize::max;
        for (usize cur = 0; cur <= allowlist.size(); cur = next) {
          next = allowlist.find(';', cur);
          if (next != string_view::npos) {
            if (allowlist.substr(cur, next - cur).compare(item) == 0) {
              return true;
            }
            next++;
          } else {
            if (allowlist.substr(cur).compare(item) == 0) {
              return true;
            }
            break;
          }
        }
        return false;
        */
}

/**
  | Returns true iff the given op name is
  | on the allowlist and should be registered
  |
  */
pub fn op_allowlist_check(op_name: StringView) -> bool {
    
    todo!();
        /*
            assert(op_name.find("::") != string_view::npos);
      // Use assert() instead of throw() due to a gcc bug. See:
      // https://stackoverflow.com/questions/34280729/throw-in-constexpr-function
      // https://github.com/fmtlib/fmt/issues/682
      assert(op_name.find("(") == string_view::npos);
    #if !defined(TORCH_OPERATOR_WHITELIST)
      // If the TORCH_OPERATOR_WHITELIST parameter is not defined,
      // all ops are to be registered
      return true;
    #else
      return op_allowlist_contains(
        C10_STRINGIZE(TORCH_OPERATOR_WHITELIST),
        // This function is majorly used for mobile selective build with
        // root operators, where the overload is included in the allowlist.
        op_name);
        // // Strip overload name (as allowlist doesn't contain overloads)
        // // Another function based on this may be added when there's usage
        // // on op names without overload.
        // OperatorNameView::parse(op_name).name);
    #endif
        */
}

/**
  | Returns true iff the given schema string
  | is on the allowlist and should be registered
  |
  */
pub fn schema_allowlist_check(schema: StringView) -> bool {
    
    todo!();
        /*
            #if defined(TORCH_FORCE_SCHEMA_REGISTRATION)
      return true;
    #else
      return op_allowlist_check(schema.substr(0, schema.find("(")));
    #endif
        */
}

/**
  | schema_allowlist_check() implicitly depends on
  | a macro, TORCH_OPERATOR_WHITELIST.
  |
  | Add this API to pass arbitrary allowlist.
  |
  */
pub fn op_allowlist_contains_name_in_schema(
        allowlist: StringView,
        schema:    StringView) -> bool {
    
    todo!();
        /*
            return op_allowlist_contains(allowlist, schema.substr(0, schema.find("(")));
        */
}

/**
  | Returns true iff the given dispatch key is on
  | the allowlist and should be registered.
  |
  | When we turn this on, the list of valid mobile
  | dispatch keys is hard coded (but you need to
  | make sure that you have the correct set of
  | dispatch keys for this).
  |
  */
pub fn dispatch_key_allowlist_check(k: DispatchKey) -> bool {
    
    todo!();
        /*
            #ifdef C10_MOBILE
      return true;
      // Disabled for now: to be enabled later!
      // return k == DispatchKey::CPU || k == DispatchKey::Vulkan || k == DispatchKey::QuantizedCPU || k == DispatchKey::BackendSelect || k == DispatchKey::CatchAll;
    #else
      return true;
    #endif
        */
}
