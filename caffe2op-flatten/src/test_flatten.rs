crate::ix!();

#[test] fn flatten_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Flatten",
        ["X"],
        ["Y"],
        axis=1
    )

    workspace.FeedBlob("X", np.random.rand(1,3,2,2))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X: [[[[0.53432311 0.23734561]
       [0.56481598 0.52152617]]

      [[0.33662627 0.32472711]
       [0.17939016 0.97175851]]

      [[0.87226421 0.49045439]
       [0.92470531 0.30935077]]]]
    Y: [[0.53432311 0.23734561 0.56481598 0.52152617 0.33662627 0.32472711
      0.17939016 0.97175851 0.87226421 0.49045439 0.92470531 0.30935077]]
    */
}
