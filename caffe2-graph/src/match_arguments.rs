crate::ix!();

/**
  | This ensures that each named arg that
  | exists in the pattern exists in g_op,
  | is equal in value.
  |
  */
#[inline] pub fn match_arguments(
    p_op: &OperatorDef,
    g_op: &OperatorDef) -> bool 
{
    todo!();
    /*
        for (const auto& p_arg : p_op.arg()) {
        if (!p_arg.has_name()) {
          continue;
        }
        bool found = false;
        for (const auto& g_arg : g_op.arg()) {
          if (p_arg.name() == g_arg.name()) {
            found = true;
            if (p_arg.has_f()) {
              if (!g_arg.has_f() || p_arg.f() != g_arg.f()) {
                return false;
              }
            }
            if (p_arg.has_i()) {
              if (!g_arg.has_i() || p_arg.i() != g_arg.i()) {
                return false;
              }
            }
            if (p_arg.has_s()) {
              if (!g_arg.has_s() || !MatchStrings(p_arg.s(), g_arg.s())) {
                return false;
              }
            }
            if (p_arg.floats_size() != g_arg.floats_size()) {
              return false;
            }
            for (int i = 0; i < p_arg.floats_size(); i++) {
              if (p_arg.floats(i) != g_arg.floats(i)) {
                return false;
              }
            }
            if (p_arg.ints_size() != g_arg.ints_size()) {
              return false;
            }
            for (int i = 0; i < p_arg.ints_size(); i++) {
              if (p_arg.ints(i) != g_arg.ints(i)) {
                return false;
              }
            }
            if (p_arg.strings_size() != g_arg.strings_size()) {
              return false;
            }
            for (int i = 0; i < p_arg.strings_size(); i++) {
              if (!MatchStrings(p_arg.strings(i), g_arg.strings(i))) {
                return false;
              }
            }
          }
        }
        if (!found) {
          return false;
        }
      }
      return true;
    */
}
