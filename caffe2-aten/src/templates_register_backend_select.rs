crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/templates/RegisterBackendSelect.cpp]

/**
  | We register ops with a higher priority dispatch
  | key (BackendSelect) than the usual
  | backend-specific keys (e.g. CPU) which makes
  | calls to the factory functions dispatch to
  | here.
  |
  | We then 'manually' compute a lower-priority to
  | re-dispatch to (e.g. CPU) to get to the
  | eventually correct
  | backend. ${generated_comment}
  */
lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(aten, BackendSelect, m) {
      ${backend_select_function_registrations};
    }
    */
}
