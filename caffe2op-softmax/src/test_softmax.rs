crate::ix!();

#[test] fn softmax_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Softmax",
        ["X"],
        ["Y"]
    )

    workspace.FeedBlob("X", np.random.randn(1, 5).astype(np.float32))
    print("input:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("softmax:", workspace.FetchBlob("Y"))

    ```

    **Result**

    ```
    input: [[ 0.0417839   0.61960053 -0.23150268 -0.64389366 -3.0000346 ]]
    softmax: [[0.24422921 0.43525138 0.18582782 0.12303016 0.01166145]]
    */
}

#[test] fn softmax_example_op() {

    todo!();

    /*
    Applies the Softmax function to an n-dimensional input Tensor rescaling them so
    that the elements of the n-dimensional output Tensor lie in the range (0,1) and
    sum to 1. The softmax operator is typically the last layer in a classifier network,
    as its output can be interpreted as confidence probabilities of an input belonging
    to each class. The input is a 2-D tensor (Tensor) of size (batch_size x
    input_feature_dimensions). The output tensor has the same shape and contains the
    softmax normalized values of the corresponding input. The softmax function is
    defined as follows:

    $$softmax(x_i) = \frac{\exp(x_i)}{\sum_{j} \exp(x_j)}$$

    The input does not need to explicitly be a 2D vector; rather, it will be coerced
    into one. For an arbitrary n-dimensional tensor `X` in
    $[a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}]$, where k is the `axis` provided,
    then `X` will be coerced into a 2-dimensional tensor with dimensions
    $[(a_0 * ... * a_{k-1}), (a_k * ... * a_{n-1})]$. For the default case where
    `axis`=1, the `X` tensor will be coerced into a 2D tensor of dimensions
    $[a_0, (a_1 * ... * a_{n-1})]$, where $a_0$ is often the batch size. In this
    situation, we must have $a_0 = N$ and $a_1 * ... * a_{n-1} = D$. Each of these
    dimensions must be matched correctly, or else the operator will throw errors.

    Github Links:

    - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softmax_op.h
    - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softmax_op.cc
    */
}

#[test] fn softmax_with_loss_op_example1() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "SoftmaxWithLoss",
        ["logits", "labels"],
        ["softmax", "avgloss"]
    )

    workspace.FeedBlob("logits", np.random.randn(1, 5).astype(np.float32))
    workspace.FeedBlob("labels", np.asarray([4]).astype(np.int32))
    print("logits:", workspace.FetchBlob("logits"))
    print("labels:", workspace.FetchBlob("labels"))
    workspace.RunOperatorOnce(op)
    print("softmax:", workspace.FetchBlob("softmax"))
    print("avgloss:", workspace.FetchBlob("avgloss"))

    logits: [[-0.3429451  -0.80375195  0.23104447  1.4569176  -0.5268362 ]]
    labels: [4]
    softmax: [[0.09721052 0.0613179  0.17258129 0.58800864 0.0808817 ]]
    avgloss: 2.5147676

    */
}

#[test] fn softmax_with_loss_op_example2() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "SoftmaxWithLoss",
        ["logits", "labels"],
        ["softmax", "avgloss"],
        scale=5.0
    )

    workspace.FeedBlob("logits", np.asarray([[.1, .4, .7, 1.5, .2]]).astype(np.float32))
    workspace.FeedBlob("labels", np.asarray([4]).astype(np.int32))
    print("logits:", workspace.FetchBlob("logits"))
    print("labels:", workspace.FetchBlob("labels"))
    workspace.RunOperatorOnce(op)
    print("softmax:", workspace.FetchBlob("softmax"))
    print("avgloss:", workspace.FetchBlob("avgloss"))

    logits: [[0.1 0.4 0.7 1.5 0.2]]
    labels: [4]
    softmax: [[0.10715417 0.144643   0.19524762 0.4345316  0.11842369]]
    avgloss: 10.667433

    */
}
