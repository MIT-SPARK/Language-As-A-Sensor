"""Label sets that decide which VLA-3D statements are used for training and evaluation."""

VALID_RELATIONS: frozenset[str] = frozenset({"on", "in", "near", "above", "below", "between"})

# NYU40 categories a statement's target and anchors must belong to.
#
# The last four (otherprop, otherfurniture, otherstructure, person) are the
# labels the released LSM checkpoint was trained with. Keeping them is what
# makes `build_splits` reproduce the published partition (1,399,294 train
# statements); without them the corpus is 2.46x smaller and most of val_seen
# would overlap the checkpoint's training data.
VALID_NYU40_LABELS: frozenset[str] = frozenset({
    "cabinet", "bed", "chair", "sofa", "table", "bookshelf", "picture", "counter",
    "blinds", "desk", "shelves", "curtain", "dresser", "pillow", "mirror", "floormat",
    "clothes", "books", "refrigerator", "television", "paper", "towel", "showercurtain",
    "box", "whiteboard", "nightstand", "toilet", "sink", "lamp", "bathtub", "bag",
    "otherprop", "otherfurniture", "otherstructure", "person",
})

# Catch-all NYU40 bins. For these the raw label is more informative, so it is
# used when counting ambiguity and when describing objects to an LLM.
NYU40_CATCHALL: frozenset[str] = frozenset({"otherprop", "otherfurniture", "otherstructure"})


def semantic_label(metadata: dict) -> str:
    """The label an object is known by: NYU40, or its raw label for catch-all bins."""
    nyu40 = metadata.get("nyu40_label") or ""
    if nyu40 in NYU40_CATCHALL:
        return metadata.get("raw_label") or nyu40
    return nyu40 or metadata.get("raw_label") or ""
