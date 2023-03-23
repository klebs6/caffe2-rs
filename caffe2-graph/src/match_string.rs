crate::ix!();

/**
  | This allows for the use of * and | to match
  | operator types, engines, or any other
  | property that is represented by strings.
  | 
  | For example, if we wanted to match an
  | operator to Conv or FC, we can give: "Conv|FC"
  | as the type() of that op.
  |
  */
#[inline] pub fn match_strings(
    p: String,
    s: String) -> bool 
{
    todo!();
    /*
        if (p == "*") { // star accepts anything
        return true;
      }
      // TODO(benz): memoize this. (high constant factor boost in performance)
      vector<string> choices = split('|', p);
      for (const string& candidate : choices) {
        if (candidate == s) {
          return true;
        }
      }
      return false;
    */
}
