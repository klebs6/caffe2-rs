crate::ix!();

/**
  | Determine which FPN level each RoI in
  | a set of RoIs should map to based on the 
  | heuristic in the FPN paper.
  |
  */
#[inline] pub fn map_ro_is_to_fpn_levels(
    rois:            &ERArrXXf,
    k_min:           f32,
    k_max:           f32,
    s0:              f32,
    lvl0:            f32,
    legacy_plus_one: Option<bool>) -> ERArrXXf {

    let legacy_plus_one: bool = legacy_plus_one.unwrap_or(false);
    
    todo!();
    /*
        // Compute level ids
      ERArrXXf s = BoxesArea(rois, legacy_plus_one).sqrt();
      // s0 = cfg.FPN.ROI_CANONICAL_SCALE  # default: 224
      // lvl0 = cfg.FPN.ROI_CANONICAL_LEVEL  # default: 4

      // Eqn.(1) in FPN paper
      // equivalent to python code
      //   target_lvls = np.floor(lvl0 + np.log2(s / s0 + 1e-6))
      //   target_lvls = np.clip(target_lvls, k_min, k_max)
      auto target_lvls = (lvl0 + (s / s0 + 1e-6).log() / log(2)).floor();
      auto target_lvls_clipped = target_lvls.min(k_max).max(k_min);
      return target_lvls_clipped;
    */
}
