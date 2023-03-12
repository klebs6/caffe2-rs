crate::ix!();

register_cpu_operator!{
    CosineEmbeddingCriterion,
    CosineEmbeddingCriterionOp<CPUContext>
}

register_cpu_operator!{
    CosineEmbeddingCriterionGradient,
    CosineEmbeddingCriterionGradientOp<CPUContext>
}

register_gradient!{
    CosineEmbeddingCriterion,
    GetCosineEmbeddingCriterionGradient
}
