crate::ix!();

#[test] fn floor_op_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Floor",
        ["X"],
        ["X"],
    )

    workspace.FeedBlob("X", (np.random.uniform(-10, 10, (5,5))).astype(np.float32))
    print("X before running op:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("X after running op:", workspace.FetchBlob("X"))

    X before running op:
    [[ 3.813361   -1.319647    5.2089314  -4.931328    0.6218652 ]
     [ 7.2757645   5.5552588   5.785643   -2.4790506  -0.41400087]
     [ 1.1541046  -6.933266    3.3754056   1.6569928  -1.7670316 ]
     [-3.4932013   4.891472    1.5530115  -3.2443287  -4.605099  ]
     [-4.574543   -7.360948    5.91305    -8.196495   -5.357458  ]]
    X after running op:
    [[ 3. -2.  5. -5.  0.]
     [ 7.  5.  5. -3. -1.]
     [ 1. -7.  3.  1. -2.]
     [-4.  4.  1. -4. -5.]
     [-5. -8.  5. -9. -6.]]
    */
}
