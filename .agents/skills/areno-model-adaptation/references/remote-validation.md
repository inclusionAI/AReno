# Remote Validation

1. Develop on a local branch; do not edit the remote checkout directly.
2. Commit the branch locally. Push it only when the repository owner authorizes
   the target remote operation.
3. On the authorized host, fetch and check out or pull the exact committed
   branch in `~/AReno`.
4. If accelerated sources changed, run `pip install -e . --no-deps --no-build-isolation`. Otherwise a pull is sufficient.
5. Before each GPU command, inspect GPUs and record the remote commit.
6. Run bounded load, inference, train backward, save, and reload gates in order.
7. Run the adapted model through at least two consecutive successful end-to-end training steps. Confirm that both steps complete with finite loss, metrics, and gradients; a one-step smoke train does not satisfy this gate.
8. If a run appears stalled or emits no logs for a long time, inspect it with `py-spy dump -p <pid>` and GPU tools before changing implementation. Use the stack to distinguish Triton/Inductor compilation, Python data processing, distributed waits, and a real kernel hang.

Do not copy uncommitted local files to the remote as an implicit deployment mechanism.
Do not patch, format, or commit source code in the remote checkout. Apply every
fix locally, commit it, and pull the new commit remotely before rerunning.
