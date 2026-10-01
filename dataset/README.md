Place legally obtained, labelled source images in `dataset/raw/<class-name>/`.

Expected example folders:

- `Tomato___healthy/`
- `Tomato___Early_blight/`
- `Tomato___Late_blight/`
- `Potato___healthy/`
- `Potato___Early_blight/`
- `Potato___Late_blight/`
- `Corn_(maize)___healthy/`
- `Corn_(maize)___Common_rust_/`
- `Corn_(maize)___Northern_Leaf_Blight/`

The class list is determined from the folders actually present; it is not fixed to this example. Keep plant, capture-session, or source-level groups out of more than one split. The splitter removes byte-identical duplicate files before splitting, but it does not detect near-duplicates. Review images for repeated plants, bursts, and field leakage before training. Dataset files are intentionally excluded from Git.