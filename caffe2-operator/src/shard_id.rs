crate::ix!();

/**
  | Extract the shard id from name of the form
  | "...shard:123..."
  |
  | Return -1 if there is no shard found
  */
#[inline] pub fn extract_shard_id(name: &String) -> i32 {
    
    todo!();
    /*
        const std::string kShard = "shard:";
      // We sometimes have multiple shards, but actually need the last one, hence
      // using rfind here. Hacky but it works till we pass shard id in graph
      // metadata.
      auto pos = name.rfind(kShard);
      if (pos != std::string::npos) {
        int left_pos = pos + kShard.length();
        int right_pos = left_pos;
        while (right_pos < name.length() && isdigit(name[right_pos])) {
          right_pos++;
        }
        return c10::stoi(name.substr(left_pos, right_pos - left_pos));
      } else {
        return -1;
      }
    */
}

/**
  | Return unique shard id, or -1 if it is
  | not unique.
  |
  */
#[inline] pub fn get_unique_shard_id(op_def: &OperatorDef) -> i32 {
    
    todo!();
    /*
        int unique_shard_id = -1;
      for (const auto& names : {op_def.input(), op_def.output()}) {
        for (const auto& name : names) {
          int shard_id = extractShardId(name);
          if (shard_id != -1) {
            if (unique_shard_id != -1) {
              return -1;
            }
            unique_shard_id = shard_id;
          }
        }
      }
      return unique_shard_id;
    */
}
