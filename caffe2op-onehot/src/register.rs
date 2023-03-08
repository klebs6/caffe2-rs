crate::ix!();

register_cpu_operator!{BatchBucketOneHot,  BatchBucketOneHotOp<CPUContext>}
register_cpu_operator!{BatchOneHot,        BatchOneHotOp<CPUContext>}
register_cpu_operator!{OneHot,             OneHotOp<CPUContext>}
register_cpu_operator!{SegmentOneHot,      SegmentOneHotOp}

no_gradient!{BatchOneHot}
no_gradient!{OneHot}
no_gradient!{SegmentOneHot}
no_gradient!{BucketBatchOneHot}
