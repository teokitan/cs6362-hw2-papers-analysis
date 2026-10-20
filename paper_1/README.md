# Reproducing Figure 1(a–c): Zhang et al., ICLR 2017

An original PyTorch reproduction of Figure 1 in **Understanding Deep Learning Requires Rethinking Generalization**, written from the user-supplied paper and assignment PDFs. **No official or unofficial implementation was consulted. No empirical values from the paper are copied into any plot.** The original Figure 1 image is included solely for visual comparison and is embedded in the notebook.

## Files

- `figure1_reproduction.ipynb` — complete annotated Jupyter notebook with **all training source code** inside a `%%writefile` cell; generates the worker module automatically.
- `zhang_fig1_core.py` — optional standalone copy of the exact generated module, useful for source review and GitHub. The notebook does not require this file as an external dependency.
- `original_figure1.png` — a crop of Figure 1 from page 4 of the user-supplied paper. Embedded separately in the notebook, so uploading the PNG is optional.
- `README.md` — these instructions.
- `.gitignore` — prevents CIFAR-10 data, trained weights, and transient files from accidentally entering version control.

## What is reproduced

**Original panel (a):** Small Inception learning curves with true labels, fixed uniformly random labels, a shared spatial-pixel permutation, independently permuted spatial pixels per image, and dataset-statistics-matched Gaussian pixels. Every input perturbation is drawn once per image, not resampled on each epoch.

**Panels (b) and (c):** The Inception six-level label-corruption grid `p = 0, .2, .4, .6, .8, 1`, plus the MLP 1×512 three-level grid `p = 0, .4, 1`. AlexNet is deliberately omitted with the student's approval. Model checkpoints are reused automatically across panels (a), (b), and (c) where experiment conditions match.

**Convergence modification:** At least 99% correct on the entire 50,000-item corrupted training set, as evaluated at the end of an epoch with `model.eval()`. This is **not** synonymous with the paper's zero training error; all generated plots label this deviation explicitly. Panel (c) is clean-test error, not identically the generalization gap at 99% training accuracy. Failures to reach 99% have no fabricated convergence time or test-error curve point.

12 **distinct** model-fitting experiments: nine Inception, three MLP, one fixed seed per condition, true FP32 (TF32/AMP disabled).

## Running on free Kaggle or Colab GPUs

1. Open `figure1_reproduction.ipynb` in a **GPU** notebook runtime. Kaggle **T4 ×2** is preferred when available; Kaggle/Colab single GPUs also work. Double-GPU availability and runtimes are **not guaranteed** by either provider.
2. If using Kaggle, enable Internet at least for the initial CIFAR-10 download, or provide the original CIFAR-10 Python-format dataset in the path configured by `DATA_ROOT` (`cifar-10-batches-py`). Colab generally permits the `torchvision` download.
3. Run cells sequentially. The module source is written by a notebook cell; built-in CPU functional checks verify both parameter counts, forward/backward pass, corruption semantics and whitening before training.
4. Review the configuration cell. Defaults are `BATCH_SIZE=128`, `MAX_EPOCHS=200`, `TRAIN_ACCURACY=.99`, `MAX_WORKER_GPUS=2`, and `MAX_WALL_HOURS=5.75`. **The paper does not specify minibatch size.** Batch size and the epoch safeguard are explicitly disclosed experiment assumptions, not assertions about the authors' settings. Changing a setting or module source after producing results requires using a fresh results directory (the worker enforces settings/source consistency).
5. Run the `run_all_experiments()` cell. Two **separate processes**, one per GPU, train independent runs in parallel on Kaggle T4 ×2; the same code automatically runs sequentially with a single GPU in Colab. CIFAR-10 training and test tensors reside on the worker GPU, avoiding repeated host-to-GPU batch transfers. Each run saves checkpoints every completed epoch and can resume by rerunning the training cell.
6. Run the audit and plotting cells. They read only genuine worker-produced metric files.

**Honest compute caveat:** The 5h45m wall-clock safety guard stops/avoids launching further training near the six-hour limit, but **cannot guarantee every run reaches 99%**. The `MAX_EPOCHS` per-run guard may also leave a run censored/unfinished; it is not counted as convergence. Actual completion depends on GPU access and the unknown training time for each corruption condition. If you need additional sessions, upload or persist the **entire results folder with checkpoints**, then rerun using the same configuration and worker source. Free cloud environments may interrupt sessions before the guard.

## Results and how to verify them

Inside `figure1_results/<run>/` expect:

- `metadata.json`: experimental settings, hardware, source checksum, realized fraction of changed labels and number of model parameters.
- `epochs.csv`: complete epoch, optimizer step count, evaluated training accuracy and mean online cross-entropy.
- `loss_steps.csv`: actual minibatch losses sampled every 20 optimizer steps, used for Figure 1(a).
- `status.json`: `converged`, `time_budget_reached`, or `max_epochs_reached`, with numeric convergence time **only when valid**.
- `checkpoint.pt`: complete-epoch optimizer/model/scheduler/RNG state.
- `worker.log`: stdout and error messages for diagnosis.

When **all 12 runs** converge, the plot cell writes `figure1_results/figure1_reproduced.png`; otherwise it writes **`figure1_PARTIAL_incomplete.png`**, prominently labeling the number of valid convergences and leaving missing points blank. `figure1_results/experiment_audit.csv` contains the entire run status table. The paper's original empirical numeric values appear nowhere in the source code.

## Methodology and documentation points

- Full 50,000 CIFAR-10 training and 10,000 test images; 28×28 center crop, divide by 255, per-image whitening.
- Uniform-label replacement can draw the original label; hence the expected *changed* fraction is `0.9p`.
- SGD with momentum 0.9, initial LR 0.1 (Inception), 0.01 (MLP), epoch multiplier 0.95, no explicit regularization or data augmentation.
- Parameter-count assertions: 1,649,402 (small Inception) and 1,209,866 (MLP). Inception diagram's branch sequence and offset-only batch norm yield an exact count, although batch-norm epsilon/momentum and certain preprocessing details are not fully specified by the paper.
- At full random-label corruption, clean-test chance error is theoretically 90% (shown as a theory baseline, not as a copied measurement).
- One seed does not imply error bars or robust statistical inference. Include any incomplete runs and corrections honestly in the assignment report.
- For assignment Phases 1–3, do not examine outside reproduction code. Commit the Phase 3 version before any optional author-code comparison in Phase 4; publish your GitHub repository **only on the submission day**, per the assignment.
