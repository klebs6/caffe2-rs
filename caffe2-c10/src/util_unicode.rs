crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/Unicode.h]
//-------------------------------------------[.cpp/pytorch/c10/util/Unicode.cpp]

#[cfg(_WIN32)]
pub fn u8u16(str_: &String) -> Wstring {
    
    todo!();
        /*
            if (str.empty()) {
        return wstring();
      }
      int size_needed = MultiByteToWideChar(
          CP_UTF8, 0, str.c_str(), static_cast<int>(str.size()), NULL, 0);
      TORCH_CHECK(size_needed > 0, "Error converting the content to Unicode");
      wstring wstr(size_needed, 0);
      MultiByteToWideChar(
          CP_UTF8,
          0,
          str.c_str(),
          static_cast<int>(str.size()),
          &wstr[0],
          size_needed);
      return wstr;
        */
}

#[cfg(_WIN32)]
pub fn u16u8(wstr: &Wstring) -> String {
    
    todo!();
        /*
            if (wstr.empty()) {
        return string();
      }
      int size_needed = WideCharToMultiByte(
          CP_UTF8,
          0,
          wstr.c_str(),
          static_cast<int>(wstr.size()),
          NULL,
          0,
          NULL,
          NULL);
      TORCH_CHECK(size_needed > 0, "Error converting the content to UTF8");
      string str(size_needed, 0);
      WideCharToMultiByte(
          CP_UTF8,
          0,
          wstr.c_str(),
          static_cast<int>(wstr.size()),
          &str[0],
          size_needed,
          NULL,
          NULL);
      return str;
        */
}
