crate::ix!();

#[test] fn tile_op_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "Tile",
        ["X", "tiles", "axis"],
        ["Y"]
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(5,5)))
    workspace.FeedBlob("tiles", np.array([5]).astype(np.int32))
    workspace.FeedBlob("axis", np.array([1]).astype(np.int32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))


    X:
    [[9 1 7 1 3]
     [2 3 6 2 5]
     [0 9 2 6 4]
     [5 8 1 5 9]
     [2 0 1 3 7]]
    Y:
    [[9 1 7 1 3 9 1 7 1 3 9 1 7 1 3 9 1 7 1 3 9 1 7 1 3]
     [2 3 6 2 5 2 3 6 2 5 2 3 6 2 5 2 3 6 2 5 2 3 6 2 5]
     [0 9 2 6 4 0 9 2 6 4 0 9 2 6 4 0 9 2 6 4 0 9 2 6 4]
     [5 8 1 5 9 5 8 1 5 9 5 8 1 5 9 5 8 1 5 9 5 8 1 5 9]
     [2 0 1 3 7 2 0 1 3 7 2 0 1 3 7 2 0 1 3 7 2 0 1 3 7]]

    */
}
