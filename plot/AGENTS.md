# Plotting Instructions

## Scope and Location

- Apply these rules whenever experiment results need visual inspection or the user requests a figure for this repository.
- Use gnuplot and import only `agent/plots/lib/style.gnuplot` for shared palette styling. Every figure script owns its fonts, dimensions, inputs, output path, and terminal settings.
- Store each experiment's figures under `agent/plots/#####-<title>/`. Reuse the same five-digit sequence number as the associated experiment/history round; do not create a new experiment number just to draw its results.
- Use descriptive, matching basenames for each script and its outputs, such as `qps.gnuplot` / `qps-png.gnuplot` and `qps.eps` / `qps.png` (see "Output Formats and Directories").
- Do not create an experiment figure directory merely to document these rules.

Example directory layout:

```text
agent/plots/
  AGENTS.md
  lib/
    style.gnuplot
  #####-<title>/
    README.md
    qps.gnuplot            # renders output-eps/qps.eps
    qps-png.gnuplot        # renders output-png/qps.png
    posting-coverage.gnuplot
    posting-coverage-png.gnuplot
    output-eps/
      qps.eps
      posting-coverage.eps
    output-png/
      qps.png
      posting-coverage.png
    data/
      qps.csv
      posting-coverage.csv
```

## Output Formats and Directories

- Draw **every figure in both formats**, EPS and PNG, without being asked. Each figure therefore has two scripts:
  - `<name>.gnuplot` is the EPS script and writes `output-eps/<name>.eps`.
  - `<name>-png.gnuplot` is the PNG script and writes `output-png/<name>.png`.
- The two scripts of one figure differ only in their terminal/output block and line width. Data, axes, labels, series, and styling are identical, so the formats stay comparable.
- Keep the rendered files in `output-eps/` and `output-png/` inside the figure directory, one directory per format, never next to the scripts. Do not put any other file type into these directories.
- gnuplot cannot create an output directory. Both directories must already exist when a script runs, so keep them in the figure directory together with the figures (a generator script may create them with `mkdir`).
- EPS terminal (6.1 by 2.7 inches is the starting size; wider figures such as 7.4 by 3.8 inches are fine when the legend needs room):

  ```gnuplot
  set terminal epscairo \
      enhanced \
      color \
      font FONT_GLOBAL \
      size EPS_WIDTH, EPS_HEIGHT
  set output "output-eps/<name>.eps"
  ```

- PNG terminal: use the EPS size at 200 dpi (7.4 by 3.8 inches becomes 1480 by 760 pixels) and `fontscale 1.4`. That scale was measured against the EPS rendered by Ghostscript at 200 dpi: `fontscale 2.0` made the text about 1.45 times too large and clipped the axis labels. Use `PNG_WIDTH`, `PNG_HEIGHT` and a PNG line width (for example 2.5, against 1.8 for EPS) defined in the script:

  ```gnuplot
  set terminal pngcairo \
      enhanced \
      color \
      font FONT_GLOBAL \
      fontscale 1.4 \
      size PNG_WIDTH, PNG_HEIGHT
  set output "output-png/<name>.png"
  ```

## One Script, One Figure

- Write each individual figure in its own `.gnuplot` file (one for EPS, one for PNG). Do not use `set multiplot`, multi-panel layouts, or a shared driver that is required to render multiple figures.
- Keep different metric types in separate figures. Multiple comparable series for the same metric may share one figure, with a legend when useful.
- Do not set a figure title. Use `unset title` explicitly. A series label in a legend, such as `title "CPU"` inside a plot command, is allowed and is distinct from a figure title.
- Do not draw "early" zoom figures (a magnified initial segment of a run) unless the user asks for them. The full range is the default.
- Make each script directly runnable from its own figure directory:

  ```sh
  gnuplot qps.gnuplot          # -> output-eps/qps.eps
  gnuplot qps-png.gnuplot      # -> output-png/qps.png
  ```

- Do not require a shell wrapper, external orchestration script, command-line `-e` variables, or execution from the repository root. The script must contain its own configuration, one style load, terminal/output setup, axes, and plot command.
- The shared `style.gnuplot` and data files are allowed dependencies. "Independently runnable" means that this script is the complete rendering entry point; it does not mean copying the shared palette into every script.
- Use durable relative paths for the shared style and data whenever possible. Never depend on temporary staging paths when rendering archived figures.
- A generator script (for example `make_plots.py`) may write the many near-identical gnuplot scripts of a figure directory, but the written scripts must stand alone and stay in the directory. Keep the generator there too.

## Series Styling

- Draw **every series with a solid line** (`dashtype 1`). Never distinguish series or categories by dash, dot, or point style.
- Distinguish series and categories **by color only**.
- Use the palette **`deep_blue`**: set `arg_color_palette = "deep_blue"` before loading `style.gnuplot`. Pick colors by `COLOR_PALETTE[index]` and keep the same assignment for the same kind of series in related figures (for example the first compared run `COLOR_PALETTE[1]`, the second `[2]`, the third `[4]`, so that adjacent series stay easy to tell apart). Keep legends explicit so that color is the only key.

## Per-Figure Settings and Shared Style

- Define all fonts directly in each figure's `.gnuplot` file so the user can edit that figure without changing a shared settings file:

  ```gnuplot
  FONT_GLOBAL = "Helvetica,24"
  FONT_TITLE  = "Helvetica,24"
  FONT_XLABEL = "Helvetica,24"
  FONT_YLABEL = "Helvetica,24"
  FONT_XTICS  = "Helvetica,22"
  FONT_YTICS  = "Helvetica,22"
  FONT_KEY    = "Helvetica,22"
  ```

- Define `EPS_WIDTH`, `EPS_HEIGHT` (EPS script) or `PNG_WIDTH`, `PNG_HEIGHT` (PNG script), the input filename, and the output filename directly in the same script; keep dimensions and any per-figure changes visible in its local configuration block.
- Set `arg_color_palette = "deep_blue"` before loading `style.gnuplot`, since that file expects the palette selector to exist. It provides the ten-entry `COLOR_PALETTE` array. Use colors consistently across related figures.
- The only shared script import is the following, for a figure directory directly beneath `plots/`:

  ```gnuplot
  load "../lib/style.gnuplot"
  ```

- Define all remaining settings locally. Figure rendering must continue to work when unused example or settings files are removed from `lib/`.
- Configure the terminal inside each figure script, using its own font and size (see "Output Formats and Directories" for the EPS and PNG terminal blocks).
- Set axis labels, units, tick fonts, and legend fonts explicitly where used. `FONT_TITLE` may be defined for consistency, but figures must still use `unset title`. Keep plot statements readable and explain important transformations or column meanings with English comments.
- Make figure-specific font, size, and output changes in that figure's script. Do not silently modify the shared palette to accommodate one figure.

## Standalone Script Example

This example assumes a real `data/qps.csv` with numeric columns for completed requests and throughput in queries per second. It illustrates script structure, not measured results or a data-generation procedure. The PNG script `qps-png.gnuplot` is the same file with the PNG settings shown after it.

This is an example of the EPS script:

```gnuplot
# Run from this figure directory: gnuplot qps.gnuplot
reset

# Keep all figure-specific settings here so this file can be edited independently.
FONT_GLOBAL = "Helvetica,24"
FONT_TITLE  = "Helvetica,24"
FONT_XLABEL = "Helvetica,24"
FONT_YLABEL = "Helvetica,24"
FONT_XTICS  = "Helvetica,22"
FONT_YTICS  = "Helvetica,22"
FONT_KEY    = "Helvetica,22"

EPS_WIDTH   = 6.1
EPS_HEIGHT  = 2.7
LINE_WIDTH  = 1.8
INPUT_FILE  = "data/qps.csv"
OUTPUT_FILE = "output-eps/qps.eps"

# Only the palette is shared; its selector must exist before loading it.
arg_color_palette = "deep_blue"
load "../lib/style.gnuplot"

set terminal epscairo \
    enhanced \
    color \
    font FONT_GLOBAL \
    size EPS_WIDTH, EPS_HEIGHT
set output OUTPUT_FILE

unset title
set datafile separator ","
set xlabel "Completed requests" font FONT_XLABEL
set ylabel "Throughput (queries/s)" font FONT_YLABEL
set xtics font FONT_XTICS
set ytics font FONT_YTICS
set key font FONT_KEY

# Column 1 is the completion count; column 2 is the documented QPS metric.
# Solid line, color only: every series differs by COLOR_PALETTE[index].
plot INPUT_FILE using 1:2 with lines \
    linewidth LINE_WIDTH dashtype 1 linecolor rgb COLOR_PALETTE[1] title "Run A"

unset output
```

The PNG script replaces only the size, line width, output file, and terminal block (everything else, including the `plot` command, stays identical):

```gnuplot
# Run from this figure directory: gnuplot qps-png.gnuplot
PNG_WIDTH   = 1220            # 6.1 in x 200 dpi (the EPS example above is 6.1 x 2.7 in)
PNG_HEIGHT  = 540             # 2.7 in x 200 dpi
LINE_WIDTH  = 2.5
OUTPUT_FILE = "output-png/qps.png"

set terminal pngcairo \
    enhanced \
    color \
    font FONT_GLOBAL \
    fontscale 1.4 \
    size PNG_WIDTH, PNG_HEIGHT
set output OUTPUT_FILE
```

(The pixel size is always the EPS size in inches times 200, so the PNG keeps the layout of the EPS; for example 7.4 by 3.8 inches gives 1480 by 760 pixels.)

## Data, Reproduction, and Validation

- Preserve actual measurement provenance. Do not fabricate values, fill unknown values with zero, or silently mix cumulative and interval metrics.
- For request-count plots, state whether the X axis counts submitted or completed requests. For QPS, hit rates, and coverage, document the time window, numerator, denominator, and whether values are cumulative or interval-based.
- For interval metrics over completed requests (hit ratio, coverage, assist outcomes, cumulative throughput), use **windows of 10,000 completed requests** by default (user preference): the boundary is the last sample taken at exactly a multiple of 10,000 completed requests, and the metric is the difference of the cumulative counters between two boundaries. Keep the window size in one variable of the data-preparation script. State in the README how many audited (verified) requests a window holds, because metrics built from a small audited subset stay noisy. Gauges (row counts, queue lengths) are not interval metrics and keep one point per sample.
- Keep original measurements in their history log archive. Use a durable relative path to those inputs, or preserve small derived plotting tables under the figure directory together with the exact conversion procedure.
- Re-rendering must not require hidden preprocessing. If preprocessing is necessary to create plot-ready data, preserve its script/command and the generated table, so the figure script can still be run directly.
- Include a Korean `README.md` linking the associated history record and raw measurements. Describe the figure files (both formats and both output directories), metric definitions and units, filters or transformations, the shared style dependency, and exact rendering commands. Embed the PNG figures in the README so that they can be viewed directly.
- Stage temporary outputs and perform figure generation outside timed benchmark regions. Finalize scripts, small derived data, EPS and PNG figures, and provenance into the plotting directory together.
- When producing actual figures, run each script from its own directory with no injected variables. Check that every EPS script creates its file in `output-eps/` and every PNG script creates its file in `output-png/`, and inspect the rendered figures (the PNG is the quickest) for readable labels, clipping, legends, appropriate scales, and solid lines distinguished by color only.
- A documentation-only update does not require inventing an experiment, creating synthetic performance figures, or rerunning a benchmark. State what was checked and what was not run.
