crate::ix!();

register_cpu_operator!{
    PairWiseLoss,
    PairWiseLossOp<float, CPUContext>
}

num_inputs!{PairWiseLoss, (2,3)}

num_outputs!{PairWiseLoss, 1}

inputs!{PairWiseLoss, 
    0 => ("X",        "Input blob from the previous layer, which is almost always the result of a softmax operation; X is a 2D array of size N x 1 where N is the batch size. For more info: D. Sculley, Large Scale Learning to Rank. https://www.eecs.tufts.edu/~dsculley/papers/large-scale-rank.pdf"),
    1 => ("label",    "Blob containing the labels used to compare the input"),
    2 => ("lengths",  "Optional input blob that contains the lengths of multiple sessions. The summation of this blob must be equal to the size of blob X. If lengths blob is provided, the output blob has the same size as lengths blob, and the cross entropy is computed within each session.")
}

outputs!{PairWiseLoss, 
    0 => ("Y", "Output blob after the cross entropy computation")
}

input_tags!{
    PairWiseLossOp {
        Xvalue,
        Label,
        Lengths
    }
}

output_tags!{
    PairWiseLossOp {
        Yvalue
    }
}
