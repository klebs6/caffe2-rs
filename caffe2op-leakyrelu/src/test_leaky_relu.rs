crate::ix!();

#[test] fn leaky_relu_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "LeakyRelu",
        ["X"],
        ["Y"],
        alpha=0.01
    )

    workspace.FeedBlob("X", np.random.randn(3, 3).astype(np.float32))
    print("X:\n", workspace.FetchBlob("X"), "\n")

    workspace.RunOperatorOnce(op)
    print("Y:\n", workspace.FetchBlob("Y"))

    X:
     [[-0.91060215  0.09374836  2.1429708 ]
     [-0.748983    0.19164062 -1.5130422 ]
     [-0.29539835 -0.8530696   0.7673204 ]]

    Y:
     [[-0.00910602  0.09374836  2.1429708 ]
     [-0.00748983  0.19164062 -0.01513042]
     [-0.00295398 -0.0085307   0.7673204 ]]
    */
}
