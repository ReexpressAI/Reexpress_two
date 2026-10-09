# Reexpress two: Frequently asked questions

## Where should I store an active project?

Keep active `.sdmproject` files on your Mac's **local, internal drive**, outside
shared or cloud-synchronized folders such as iCloud Drive or Dropbox. Reexpress
two checks that the volume is local and internal when you create or open a
project, so external drives and network shares cannot host active projects.

Cloud-synchronized folders can still reside on an internal drive, so the volume
check does not detect every unsuitable location. Choose a folder that is not
synchronized or shared for editing. Reexpress two does not support multiple
users editing the same project concurrently.

## How do I make a full project backup or transfer a project to another Mac?

1. Quit Reexpress two before copying the project.
2. Copy the **entire `.sdmproject` package** using Finder or your usual file-copy
   tool. Do not copy only individual files from inside the package.
3. You may store the closed backup on an external drive or transfer it through
   cloud storage or a network share. Before opening it, copy the complete
   package to a local, internal, nonsynchronized folder on the destination Mac.

A full project copy includes its imported data, models, scores, training
history, and staged review changes. Ordinary full-package copies are
independent: Editing one copy does not change the other.

## Why can't I open a different copy of the same project in the current app session?

A copied project retains the original project's identity. During one app
session, Reexpress two associates that identity with the first location opened.
To work with another copy, **quit Reexpress two, reopen the app, and open the
desired copy**. Renaming the package does not change its internal identity, and
you do not need to edit the package's contents.

## Do the same storage restrictions apply to model and dataset bundles?

No. The local, internal-drive restriction applies to **active `.sdmproject`
files**. Model (`.sdmkitmodel`) and dataset (`.sdmdataset`) bundles are separate
import/export formats and do not have that restriction, subject to ordinary
file-access permissions and availability.

Use model and dataset exports to exchange artifacts with the Python package or
other projects. Use a full `.sdmproject` copy when you want to preserve the
complete project for use in Reexpress two.

## How large can the training/calibration set be given my available compute?

The standalone [training capacity calculator](https://github.com/ReexpressAI/reexpress_sdm/blob/main/docs/tools/README.md) estimates memory ranges and approximate row counts for Reexpress two and the Python package reexpress_sdm. That script itself uses only Python's standard library, separates CUDA system/GPU memory, and reports its assumptions. The script provides planning estimates, not guarantees, and other factors (including, but not limited to, other processes running at the same time) can impact the achievable capacity in practice.

## Should the training and calibration sets be the same data I use for post-training my model?

No. The training and calibration sets for the SDM activation and SDM estimator are splits of held-out data that you would otherwise use with alternative calibration methods. They should not be the exact same data that was used to update the weights of the underlying language model (LM) producing the frozen feature vectors used with Reexpress two (or the reexpress_sdm Python package).

Relatedly, the SDM activation used with an SDM estimator for calibration is, by design, trained with frozen input features. In other words, the gradients do not flow into the underlying network below the SDM activation. That is part of the method, as opposed to a simplification for efficiency. Training the SDM activation learns a distilled, compressed representation of the underlying network's representation space conditional on its predictions, with the exemplar vector constructed from the first of the two affine transforms of the SDM activation. (The first affine operations can be equivalently viewed as a 1-D CNN.)

Reexpress two (and the reexpress_sdm Python package) do not currently support updating the underlying weights of an LM. We may provide software for such post-training in the future. Feel free to contact us if you have questions about this for your particular use-case, and/or need assistance. The key thing to keep in mind is that the updating of the LM's weights is separate from the calibration step. The general recipe to take is the following straightforward process: After post-training your LM (with whatever method you choose, via supervised fine-tuning, reinforcement learning, or related), freeze the weights, and then with a separate held-out set (itself split into a training and calibration set, which themselves get randomly shuffled by default by our software) train and calibrate the final-layer SDM activation with Reexpress two (or the reexpress_sdm Python package) prior to deployment.
