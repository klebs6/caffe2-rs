crate::ix!();

pub type TensorTuple = (Tensor, Tensor);

pub trait FullBidirectionalLSTMLayerTypes {
    type BidirHiddenType = (TensorTuple, TensorTuple);
    type ParamType       = (CellParams, CellParams);
    type OutputType      = LayerOutput<Tensor, Self::BidirHiddenType>;
}

pub trait Layer<HiddenType, ParamType> {

    type OutputType = LayerOutput::<Tensor, HiddenType>;

    fn invoke(&self, input: &Tensor, input_hidden: &HiddenType, params: &ParamType) -> Self::OutputType;
}
