crate::ix!();

#[test] fn constant_fill_example1() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ConstantFill",
        [],
        ["Y"],
        shape=(1,5,5)
    )

    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    **Result**

    Y: [[[0. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0.]
      [0. 0. 0. 0. 0.]]]
    */
}

#[test] fn constant_fill_example2() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ConstantFill",
        ["X"],
        ["Y"],
        value=4.0,
        dtype=1,
        extra_shape=(1,2)
    )

    workspace.FeedBlob("X", (np.random.randint(100, size=(3,3))).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X: [[86. 30. 84.]
     [34. 51.  9.]
     [29. 86. 59.]]
    Y: [[[[4. 4.]]

      [[4. 4.]]

      [[4. 4.]]]


     [[[4. 4.]]

      [[4. 4.]]

      [[4. 4.]]]


     [[[4. 4.]]

      [[4. 4.]]

      [[4. 4.]]]]

    */
}
