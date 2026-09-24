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
