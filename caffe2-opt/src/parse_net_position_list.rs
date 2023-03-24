crate::ix!();

/**
  | The list in in the form of "0-3,5,6-7"
  | which means, we will black list ops with
  | net positions in [0,1,2,3,5,6,7]
  |
  */
#[inline] pub fn parse_net_position_list(str: &String) -> HashSet<i32> {
    
    todo!();
    /*
        std::unordered_set<int> net_position_list;
      if (str.empty()) {
        return net_position_list;
      }
      auto tokens = caffe2::split(',', str);
      for (const auto& token : tokens) {
        if (token == "-1") {
          net_position_list.emplace(-1);
          continue;
        }
        auto range = caffe2::split('-', token);
        if (range.size() == 1) {
          net_position_list.emplace(std::stoi(range[0]));
        } else if (range.size() == 2) {
          int from = std::stoi(range[0]);
          int to = std::stoi(range[1]);
          for (int i = from; i <= to; ++i) {
            net_position_list.emplace(i);
          }
        } else if (range.size() > 2) {
          LOG(WARNING) << "Ignoring illegal range: " << token;
        }
      }
      return net_position_list;
    */
}


