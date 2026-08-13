# GPU container recipe

Build the optional LeRobot/SmolVLA image from the repository root:

```bash
docker compose -f deploy/compose.yaml build
docker compose -f deploy/compose.yaml run --rm fleetvla demo
```

The image installs the policy extra only. LIBERO additionally needs its EGL and
MuJoCo system libraries, so use the documented host install for
`fleetvla libero` until a simulator-specific image is published; the SmolVLA
image is not presented as a LIBERO image.

The demo is still CPU-only; it verifies the package inside the image. Real
model use requires the NVIDIA Container Toolkit, compatible host drivers, and a
pinned model revision. The `fleetvla libero` command constructs the reference
SmolVLA/LIBERO path on a host with the optional simulator dependencies. The
compose file persists only the Hugging Face cache.
Pass tokens at runtime through your secret manager; never bake them into the
image or commit them.

Record GPU model, driver, CUDA/PyTorch/LeRobot versions, model revision, warmup,
precision, batch limits, and measured latency profile with results. The
synthetic backend's modeled times are not GPU performance claims.
