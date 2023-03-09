crate::ix!();

#[test] fn add_padding_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "AddPadding",
        ["X", "lengths"],
        ["Y", "lengths_out"],
        padding_width=1

    )

    workspace.FeedBlob("X", (np.random.rand(3,2,2).astype(np.float32)))
    workspace.FeedBlob("lengths", np.array([3]).astype(np.int32))

    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))
    print("lengths_out:", workspace.FetchBlob("lengths_out"))

    X: [[[0.2531572  0.4588472 ]
      [0.45140603 0.61161053]]

     [[0.92500854 0.8045306 ]
      [0.03356671 0.30233648]]

     [[0.4660227  0.6287745 ]
      [0.79372746 0.08609265]]]
    Y: [[[0.         0.        ]
      [0.         0.        ]]

     [[0.2531572  0.4588472 ]
      [0.45140603 0.61161053]]

     [[0.92500854 0.8045306 ]
      [0.03356671 0.30233648]]

     [[0.4660227  0.6287745 ]
      [0.79372746 0.08609265]]

     [[0.         0.        ]
      [0.         0.        ]]]
    lengths_out: [5]
    */
}
