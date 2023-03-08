crate::ix!();

#[inline] pub fn get_recurrent_mapping(
    links: &Vec<Link>,
    backward: bool) -> HashMap<String,String> 
{
    todo!();
    /*
        std::map<string, string> mappings;
      for (auto it = links.begin(); it != links.end(); ++it) {
        const auto& l1 = *it;

        // In backward op we expect to see offset 1 before offset 0 and
        // vice versa.
        const int offset_l1 = backward ? 1 : 0;
        const int offset_l2 = 1 - offset_l1;
        if (l1.offset == offset_l1) {
          // Find offset = 1 from links. We could probaby rely on order, but
          // since the number of links is links small, O(n^2) algo is ok
          for (auto it2 = it + 1; it2 != links.end(); ++it2) {
            const auto& l2 = *it2;
            if (l2.offset == offset_l2 && l2.external == l1.external) {
              mappings[l2.internal] = l1.internal;
              break;
            }
          }
        }
      }
      return mappings;
    */
}
