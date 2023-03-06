crate::ix!();

#[inline] pub fn klog_threshold() -> f32 {
    1e-20
}

#[inline] pub fn logit(p: f32) -> f32 {
    
    todo!();
    /*
      // it computes log(p / (1-p))
      // to avoid numeric issue, hard code p log(p) when p approaches 0
      float x = std::min(std::max(p, kLOG_THRESHOLD()), 1 - kLOG_THRESHOLD());
      return -log(1. / x - 1.);
    */
}

#[inline] pub fn entropy(p: f32) -> f32 {
    
    todo!();
    /*
      if (p < kLOG_THRESHOLD() || 1 - p < kLOG_THRESHOLD()) {
        return 0.;
      } else {
        float q = 1 - p;
        return -p * log(p) - q * log(q);
      }
    */
}
