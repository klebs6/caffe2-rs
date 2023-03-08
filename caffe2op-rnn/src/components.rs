crate::ix!();

pub struct Param {
    param:               String,
    grad:                String,
    cell_gradient:       String,
}

pub struct RecurrentInput {
    state:               String,
    input:               String,
}

pub struct RecurrentGradient {
    param:               String,
    grad:                String,
    external_grad:       String,
    last_external_grad:  String,
    offset:              i32,
}

pub struct OffsetAlias {
    src:                 String,
    dst:                 String,
    offset:              i32, //{0};
}

pub struct Link {
    internal:            String,
    external:            String,
    offset:              i32,//{0};
    window:              i32,//{1};
}

pub struct ScratchWorkspaces {
    step_workspaces:     Vec<Arc<Workspace>>,
    shared_blobs_ws:     Arc<Workspace>,
}
