crate::ix!();

/**
  | dynamic cast reroute: if RTTI is disabled,
  | go to reinterpret_cast
  |
  */
#[inline] pub fn dynamic_cast_if_rtti<Dst, Src>(ptr: Src) -> Dst {
    todo!();
    /*
        #ifdef __GXX_RTTI
      return dynamic_cast<Dst>(ptr);
    #else
      return static_cast<Dst>(ptr);
    #endif
    */
}
