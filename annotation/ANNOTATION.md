# Annotating the plant marked by each cardboard

Only the plant is annotated. The cardboard markers are not annotation objects. They stay
visible in the imagery as a reference the annotator reads by eye.

## Layout

```
raw/                     62 source stills, 8192 x 5460
processed/
  batch_01 .. batch_07   620 tiles, 100 each except batch_07 which holds 20
  cvat_labels.json       label schema for task creation, plant only
  tile_markers.csv       which board sits in which tile, with uid and arrow colour
  manifest.csv           all 2976 tiles considered, window, kept flag, marker px, batch
  markers.csv            548 markers in full-image coords
  markers_classified.csv the same markers plus arrow_colour
tile_images.py           tiling and marker detection
make_cvat_task.py        arrow classification, label schema, crib sheet
split_batches.py         batching
```

Regenerate with the following commands.

```sh
.venv/bin/python tile_images.py --input raw --output processed --fresh
.venv/bin/python make_cvat_task.py
.venv/bin/python split_batches.py
```

## Workflow

1. Create one CVAT task per batch folder, using `cvat_labels.json` as the label schema.
   The only label is `plant`, drawn as a polygon.
2. For each cardboard visible in the tile, read its arrow to identify the target plant.
3. Prompt SAM3 with a click or box on that plant to generate the canopy polygon, then
   accept or correct the mask.
4. Set `class` from the arrow colour, tick `partial` when the canopy runs off the tile
   edge, and copy the board's `marker_uid` from `tile_markers.csv`.

`marker_uid` is formed as source stem plus blob id. It is what pairs a plant back to the
board that marked it, across tiles and back to full-image coordinates. It is the only
remaining link to the markers now that they are not annotated, so it matters.

If you later want the boards exported as rectangles after all, run
`make_cvat_task.py --include-markers`, which restores the pre-annotation XML.

## Why it is set up this way

**Greenness does not find these plants.** An excess green index on image 0185 returns 125
plant-sized blobs, and nearest-blob-to-marker lands on small grass tufts every time. In
this dry-season set the target plants are the brown, leafless, thorny shrubs, so a
vegetation index points away from them. There is no reliable automatic nearest plant here.

**Nearest is ambiguous on its own.** The scrub is near continuous, so a pure distance rule
frequently picks a neighbouring bush. The arrow is the actual ground truth for which plant
was meant, which is why the workflow is arrow led rather than distance led.

**Arrow direction is not automated.** Both heuristics tried were around chance at this
ground sample distance, so the annotator reads the arrow by eye. See the module docstring
in `make_cvat_task.py`.

## Known issues to resolve before annotating

- **Blank boards, 40 confirmed.** Some boards carry no arrow at all. Decide what they mean,
  whether a third class, a flipped board, or a non-plant marker such as a GCP. They have no
  direction, so they cannot be resolved by the arrow rule.
- **Unknown colour, 133 markers or 24%.** Small or partly occluded boards where no paint was
  detected. These need an eye on them. The value is deliberately `unknown` rather than a guess.
- **Duplicate boards in the overlap band.** `--edge shift` makes the bottom tile row overlap
  the row above by 684 px, so a board there appears in two tiles. The 548 distinct markers
  produce 737 sightings in `tile_markers.csv`. De-duplicate on `marker_uid` before computing
  dataset statistics, otherwise those plants are double counted.
- **Boards split across a tile boundary** appear in two tiles under one `marker_uid`, for the
  same reason.
- **Marker detection is recall biased.** 72 of the 620 tiles, which is 11.6%, rest only on a
  weak detection under 1500 px, and a minority of those are bright grass rather than cardboard.
  Tiles with no cardboard in them can simply be skipped.
