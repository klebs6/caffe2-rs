crate::ix!();

pub struct WorkspaceBookkeeper {
    wsmutex:    parking_lot::RawMutex,
    workspaces: HashSet<Arc<Workspace>>,
}

lazy_static!{
    static ref bookkeeper: Arc<WorkspaceBookkeeper> = Arc::new(WorkspaceBookkeeper::default());
}

impl Default for WorkspaceBookkeeper {

    fn default() -> Self {
        todo!();
    }
}
