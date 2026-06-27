# Few-shot danger anchors

Drop a handful of example frames here to define what "dangerous" vs "safe"
looks like. The generative future-preview track embeds the imagined future
frames with CLIP and scores danger by similarity to these anchors.

```
anchors/
  dangerous/   # e.g. knife touching a hand, hand over a flame, robot mid-collision
  safe/        # e.g. normal cooking, robot idle, tidy counter
```

A few images per folder (5-15) is enough. JPG or PNG.
