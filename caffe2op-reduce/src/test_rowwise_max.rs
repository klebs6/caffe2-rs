crate::ix!();

#[test] fn rowwise_max_op_example() {
    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "RowwiseMax",
        ["X"],
        ["Y"]
    )

    // Create X, simulating a batch of 2, 4x4 matricies
    X = np.random.randint(0,high=20,size=(2,4,4))
    print("X:\n",X)

    // Feed X into workspace
    workspace.FeedBlob("X", X.astype(np.float32))

    // Run op
    workspace.RunOperatorOnce(op)

    // Collect Output
    print("Y:\n", workspace.FetchBlob("Y"))

    X:
     [[[ 5 12 10  1]
      [ 4 16  2 15]
      [ 5 11 12 15]
      [15  4 17 19]]

     [[16  5  5 13]
      [17  2  1 17]
      [18  3 19  5]
      [14 16 10 16]]]
    Y:
     [[12. 16. 15. 19.]
     [16. 17. 19. 16.]]
    */
}
