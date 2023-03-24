crate::ix!();

/**
  | A struct that holds the gradient operators
  | and related gradient maps.
  |
  */
#[derive(Default)]
pub struct GradientOpsMeta {
    ops:     Vec<OperatorDef>,
    g_input: Vec<GradientWrapper>,
}

