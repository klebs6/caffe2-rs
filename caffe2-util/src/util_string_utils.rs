crate::ix!();

#[inline] pub fn starts_with(
    str_:   &String,
    prefix: &String) -> bool {
    
    todo!();
    /*
        return str.length() >= prefix.length() &&
          std::mismatch(prefix.begin(), prefix.end(), str.begin()).first ==
          prefix.end();
    */
}

#[inline] pub fn ends_with(full: &String, ending: &String) -> bool {
    
    todo!();
    /*
        if (full.length() >= ending.length()) {
        return (
            0 ==
            full.compare(full.length() - ending.length(), ending.length(), ending));
      } else {
        return false;
      }
    */
}

#[inline] pub fn split(
    separator:    u8,
    string:       &String,
    ignore_empty: Option<bool>) -> Vec<String> 
{
    let ignore_empty = ignore_empty.unwrap_or(false);

    todo!();
    /*
        std::vector<std::string> pieces;
      std::stringstream ss(string);
      std::string item;
      while (getline(ss, item, separator)) {
        if (!ignore_empty || !item.empty()) {
          pieces.push_back(std::move(item));
        }
      }
      return pieces;
    */
}

#[inline] pub fn trim(str: &String) -> String {
    
    todo!();
    /*
        size_t left = str.find_first_not_of(' ');
      if (left == std::string::npos) {
        return str;
      }
      size_t right = str.find_last_not_of(' ');
      return str.substr(left, (right - left + 1));
    */
}


#[inline] pub fn edit_distance(
    s1: &String,
    s2: &String,
    max_distance: usize) -> usize 
{
    todo!();
    /*
        std::vector<size_t> current(s1.length() + 1);
        std::vector<size_t> previous(s1.length() + 1);
        std::vector<size_t> previous1(s1.length() + 1);

        return editDistanceHelper(
            s1.c_str(),
            s1.length(),
            s2.c_str(),
            s2.length(),
            current,
            previous,
            previous1,
            max_distance
        );
    */
}

#[macro_export] macro_rules! NEXT_UNSAFE {
    ($s:ident, $i:ident, $c:ident) =>  {
        /*
        (c)=(uint8_t)(s)[(i)++]; \
        */
    }
}

#[inline] pub fn edit_distance_helper(
    s1:            *const u8,
    s1_len:        usize,
    s2:            *const u8,
    s2_len:        usize,
    current:       &mut Vec<usize>,
    previous:      &mut Vec<usize>,
    previous1:     &mut Vec<usize>,
    max_distance:  usize) -> i32 
{
    todo!();
    /*
        if (max_distance) {
          if (std::max(s1_len, s2_len) - std::min(s1_len, s2_len) > max_distance) {
            return max_distance+1;
          }
        }

        for (size_t j = 0; j <= s1_len; ++j) {
          current[j] = j;
        }

        int32_t str2_offset = 0;
        char prev2 = 0;
        for (size_t i = 1; i <= s2_len; ++i) {
          swap(previous1, previous);
          swap(current, previous);
          current[0] = i;

          char c2 = s2[str2_offset];
          char prev1 = 0;
          int32_t str1_offset = 0;

          NEXT_UNSAFE(s2, str2_offset, c2);

          size_t current_min = s1_len;
          for (size_t j = 1; j <= s1_len; ++j) {
            size_t insertion = previous[j] + 1;
            size_t deletion = current[j - 1] + 1;
            size_t substitution = previous[j - 1];
            size_t transposition = insertion;
            char c1 = s1[str1_offset];

            NEXT_UNSAFE(s1, str1_offset, c1);

            if (c1 != c2) {
              substitution += 1;
            }


            if (prev1 == c2 && prev2 == c1 && j > 1 && i > 1) {
              transposition = previous1[j - 2] + 1;
            }
            prev1 = c1;

            current[j] = std::min(std::min(insertion, deletion),
                             std::min(substitution, transposition));
            current_min = std::min(current_min, current[j]);
          }


          if (max_distance != 0 && current_min > max_distance) {
            return max_distance+1;
          }

          prev2 = c2;
        }

        return current[s1_len];
    */
}
