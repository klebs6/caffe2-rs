crate::ix!();

pub trait BackendTransformerBaseTrait {

    fn transform(
        &mut self,
        ws: *mut Workspace,
        pred_net: *mut NetDef,
        weight_names: &Vec<String>,
        shape_hints: &ShapeInfoMap,
        blocklisted_ops: &HashSet<i32>);
}

