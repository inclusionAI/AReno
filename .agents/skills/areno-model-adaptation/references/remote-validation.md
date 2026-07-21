# Remote Validation

1. Develop on a local branch; do not edit the remote checkout directly.
2. Commit and push the branch.
3. On the authorized host, fetch and check out the exact branch in `~/AReno`.
4. If accelerated sources changed, run `pip install -e . --no-deps --no-build-isolation`. Otherwise a pull is sufficient.
5. Before each GPU command, inspect GPUs and record the remote commit.
6. Run bounded load, inference, train backward, save, and reload gates in order.
7. If a run appears stalled, inspect it with `py-spy` or GPU tools before changing implementation.

Do not copy uncommitted local files to the remote as an implicit deployment mechanism.
