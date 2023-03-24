crate::ix!();

pub const kNetPos:  &'static str = "net_pos";
pub const kModelId: &'static str = "model_id";

pub struct BackendTransformOptions {

    /**
      | Enable debugging by dumping more intermediate
      | graphs
      |
      */
    debug: bool,

    /**
      | Minimum number of ops to create a backend
      | op.
      |
      | If the subgraph is too small, it doesn't
      | make sense to lower it to backend.
      */
    min_ops: usize,

    /// Bound shape spec
    bound_shape_spec: BoundShapeSpec,
}

impl Default for BackendTransformOptions {

    fn default() -> Self {
        Self {
            debug: false,
            min_ops: 1,
            bound_shape_spec: BoundShapeSpec::new(0, 0),
        }
    }
}
