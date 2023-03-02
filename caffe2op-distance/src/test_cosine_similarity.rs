crate::ix!();

#[test] fn cosine_similarity_op_example() {
    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "CosineSimilarity",
        ["X", "Y"],
        ["Z"]
    )

    // Create X
    X = np.random.randn(3, 3)
    print("X:\n",X)

    // Create Y
    Y = np.random.randn(3, 3)
    print("Y:\n",Y)

    // Feed X & Y into workspace
    workspace.FeedBlob("X", X.astype(np.float32))
    workspace.FeedBlob("Y", Y.astype(np.float32))

    // Run op
    workspace.RunOperatorOnce(op)

    // Collect Output
    print("Z:\n", workspace.FetchBlob("Z"))

    **Result**

    X:
     [[-0.42635564 -0.23831588 -0.25515547]
     [ 1.43914719 -1.05613228  1.01717373]
     [ 0.06883105  0.33386519 -1.46648334]]
    Y:
     [[-0.90648691 -0.14241514 -1.1070837 ]
     [ 0.92152729 -0.28115511 -0.17756722]
     [-0.88394254  1.34654037 -0.80080998]]
    Z:
     [-1.7849885e-23  1.7849885e-23 -1.0842022e-07]

    */
}

