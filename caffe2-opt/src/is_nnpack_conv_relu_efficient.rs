crate::ix!();

#[inline] pub fn is_nnpackconv_relu_efficient(algo: &String, conv: &Conv) -> bool {
    
    todo!();
    /*
        if (algo == "AUTO" || algo == "") {
        for (auto stride : conv.getStrides()) {
          if (stride > 1) {
            return false;
          }
        }
        for (auto kernel : conv.getKernelShape()) {
          if (kernel < 2) {
            return false;
          }
        }
      } else if (!(algo == "WINOGRAD" || algo == "WINOGRAD_FP16" ||
                   algo == "FT8x8" || algo == "FT16x16")) {
        return false;
      }
      return true;
    */
}
