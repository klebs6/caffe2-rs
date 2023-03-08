crate::ix!();

/**
  | Range reducers: can leverage that input
  | segment is continuous and provide special
  | implementation
  |
  | Incremental reducers: consume elements
  | one by one
  |
  */
pub trait Reducer {
    const InputCount: isize;
}
