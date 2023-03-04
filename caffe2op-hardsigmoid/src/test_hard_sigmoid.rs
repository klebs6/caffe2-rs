crate::ix!();

#[test] fn hard_sigmoid_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "HardSigmoid",
        ["X"],
        ["Y"],
        alpha = 0.2,
        beta = 0.5,
    )

    workspace.FeedBlob("X", np.random.randn(5).astype(np.float32))
    print("input:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("sigmoid:", workspace.FetchBlob("Y"))

    input: [ 1.5744036   0.31632107  1.7842269   1.4450722  -2.1726978 ]
    hard_sigmoid: [ 0.81488073,  0.56326419,  0.85684538,  0.78901446,  0.06546044]
    */
}
