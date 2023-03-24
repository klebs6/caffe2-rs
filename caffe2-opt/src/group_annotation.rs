crate::ix!();

pub struct GroupAnnotation {
    group:     i32,//-1
    in_degree: i32,
    needs_transform: bool,//true
}

impl GroupAnnotation {

    pub fn new(i: i32, g: Option<i32>) -> Self {
        Self {
            group:           g.unwrap_or(-1),
            in_degree:       i,
            needs_transform: true,
        }
    }
}

