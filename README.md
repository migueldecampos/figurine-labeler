# Figurine Labeler

Label objects in your photos with a few clicks, helped by two pretrained models: OWLv2 (detection) and SAM 2.1
(segmentation). Runs entirely in your browser: nothing to install, and your photos and labels never leave your
computer. Only the tool itself and the models are downloaded (from GitHub Pages, jsDelivr and Hugging Face).
The models are downloaded the first time (about 300–800 MB depending on your computer, which can take a few
minutes) and are then normally kept by the browser. The download starts in the background as soon as the page
opens.

## Start

Open **https://migueldecampos.github.io/figurine-labeler/** in **Chrome or Edge** on a computer with a mouse or
trackpad. (Other desktop browsers work too, but can only download your labels as a zip file.)

Nothing is uploaded: you open photos (JPG or PNG) from your computer, and labels are saved back to it.

- **📁 Open a folder of photos** — the photos directly in that folder (not in subfolders) are shown. Labels are
  saved automatically into a `labels` folder inside it, so you can stop and continue later (the browser asks for
  permission to save there the first time).
  In other browsers than Chrome/Edge: click **⬇ Download labels** before closing. The zip always contains the
  complete `labels` folder: to continue later, *replace* the `labels` folder next to your photos with it
  (don't merge the two).
- **🖼 Open a single photo** — when you're done, click **⬇ Download labels** to get `<photo>-labels.zip`.
  To continue later, put the photo and the unzipped `labels` folder in a folder and open that folder.

## Label a photo

1. Pick a photo on the left (○ = not done yet, ✓ = done).
2. Click **✨ Propose**. The detection model looks for toy figurines in the photo (always, whatever classes you
   use: classes are only the names you give to objects).
   *Tip:* to focus on one object, first drag a box around it, then click Propose.
3. For each proposal (highlighted in blue): **✓ Accept**, **✗ Decline**, or **✎ Improve**.
   - **Improve** zooms in on the object. **✏️ Outline**: hold the mouse button and draw around an area to add
     it. **🖌 Paint** pixels that belong to the object, **🧽 Erase** the ones that don't. Then click **✓ Done**
     (or **✗ Cancel** to forget the changes). Zoom with a pinch or ⌘/Ctrl + scroll; scroll to move around;
     **🔍 Fit** shows the whole photo again.
4. After **Accept**, choose what it is. There are no predefined classes: the first time, create one with
   **+ New class…** (e.g. *smurf*); after that it is a one-click button. The object is saved right away.
5. If the detection model doesn't find an object, click **✏️ Draw myself** (after dragging a box around it, it zooms in
   there): outline it, fix the edges with Paint/Erase, click **✓ Done** and choose what it is.
6. Photos with nothing to label: click **Nothing to label here**.

To change something later, click a coloured object: **Improve**, **Change class**, or **Delete**.

## What gets saved

A `labels/` folder next to your photos (folder), or the same `labels/` folder inside the downloaded zip.
`<photo>` is the photo's file name without extension (e.g. `photo_1` for `photo_1.jpg`):

| File | Content |
|---|---|
| `labels/<photo>.png` | Same size as the photo, one number per pixel: `0` = background, `1`, `2`, … = objects |
| `labels/<photo>.json` | Class of each object: `{"objects": [{"id": 1, "class": "smurf"}, {"id": 2, "class": "troll"}]}` |
| `labels/classes.json` | All class names used in this folder |

How you turn this into training targets is up to you, for example a mask of all smurfs:

```python
import json, numpy as np
from PIL import Image
objects = np.array(Image.open("labels/photo_1.png"))               # (H, W) object numbers
info = json.load(open("labels/photo_1.json"))["objects"]
smurf = np.isin(objects, [o["id"] for o in info if o["class"] == "smurf"])  # (H, W) True/False
```

**Rotated photos:** phones often store photos sideways together with a "rotate me" note, which software handles
inconsistently. The labeler asks you to make such photos upright (one click) before labeling, so that photo and
labels always line up, whatever software you use later. In a folder (Chrome/Edge) this replaces the photo file
with an upright copy; otherwise you download the upright copy and keep it instead of the original.

**Limits:** photos are edited at up to 2048 pixels on the long side (labels are still saved at full size), and a
photo can have at most 255 objects.

## Run it locally

The tool is two static files (`index.html`, `app.js`) and needs no build step. Browsers only allow folder access
and some model features on `https://` or `localhost` pages, so serve the folder instead of opening the file directly:

```
python3 -m http.server 8000
```

then open http://localhost:8000.

## Credits and license

- [SAM 2.1](https://huggingface.co/facebook/sam2.1-hiera-tiny) (Meta, Apache-2.0) for segmentation, and
  [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) (Google, Apache-2.0) for detection, both via
  the ONNX conversions [sam2.1-hiera-tiny-ONNX](https://huggingface.co/onnx-community/sam2.1-hiera-tiny-ONNX) and
  [owlv2-base-patch16-ensemble-ONNX](https://huggingface.co/onnx-community/owlv2-base-patch16-ensemble-ONNX) by
  onnx-community (pinned to fixed revisions in `app.js`).
- [Transformers.js](https://github.com/huggingface/transformers.js) (Apache-2.0) and
  [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (MIT) run the models in the browser;
  [fflate](https://github.com/101arrowz/fflate) (MIT) writes PNG and zip files.

The models and libraries are downloaded by your browser from Hugging Face and jsDelivr; they are not part of
this repository.

This tool is released under the [MIT License](LICENSE).
