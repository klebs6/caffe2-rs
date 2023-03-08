crate::ix!();

#[test] fn colwise_max_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ColwiseMax",
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

    ```

    **Result**

    ```

    X:
     [[[17 15  2  6]
      [ 8 12  6  0]
      [ 6  9  7  3]
      [ 4 13 16 13]]

     [[ 0  3  4 12]
      [18  1 17 12]
      [ 7 17 13 14]
      [12 17  2  1]]]
    Y:
     [[17. 15. 16. 13.]
     [18. 17. 17. 14.]]
    */
}
