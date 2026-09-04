# Using the scanner on a real product

## One-time setup

Follow the **Setup** section of [`README.md`](README.md) — clone, create the virtual
environment, activate it, install Tesseract, then:

```bash
python backend/verify.py     # should print 11/11 checks passed
```

Every command on this page assumes your virtual environment is **activated**
(`source .venv/bin/activate` on macOS/Linux, `.venv\Scripts\Activate.ps1` on Windows).
On Windows, replace `open <file>` with `start <file>`.

## One-time: print and measure the calibration card

```bash
open backend/assets/calibration_card_A4_300dpi.pdf
```

Print it at **100% scale** — turn OFF "fit to page" / "scale to fit". Printers shrink
pages by default and that silently corrupts every measurement.

Then **measure the printed black square** edge to edge with a ruler or calipers. The
card also carries a 100 mm scale bar; check it reads 100 mm. Whatever you measure is
what you pass as `--marker-mm`. If it measures 39.4 mm, pass `39.4`, not `40`.

This number is the only thing connecting pixels to millimetres. Getting it wrong scales
every result by the same error.

## Photographing a product

1. Pick a **flat** face of the pack — a carton panel, a flat pouch face, a flat label.
   Bottles, jars and cans are out of scope for this build.
2. Lay the card **flat against that same face**, right next to the declarations. The
   card and the printed text must be in the **same plane**. A card lying on the table
   next to a standing box gives a wrong answer that looks confident.
3. **Fill the frame.** The label plus the card should occupy most of the photo width.
   Do not stand back and crop later, and do not use digital zoom — move the camera
   closer. Shoot at your phone's full resolution.
4. Shoot as **square-on** as you can. Slight tilt is corrected automatically; past
   ~25–30° accuracy degrades and the scan is refused.
5. Avoid a specular highlight sitting on the marker or on the small print. Diffuse
   light (near a window, or indoor light off to one side) is better than direct flash.

**Why "fill the frame" matters:** the system needs about 16 pixels per millimetre for
its best accuracy. A 12 MP phone is ~4000 px wide, so a frame covering 25 cm gives
16 px/mm. A frame covering 60 cm gives only 7 px/mm, where the error is roughly four
times worse.

## Scanning a photo

```bash
python backend/lmpc/cli.py scan ~/Desktop/product.jpg --marker-mm 39.4
```

Useful flags:

| flag | purpose |
|---|---|
| `--marker-mm 39.4` | **required** — your measured marker size |
| `--save` | store the scan in the SQLite history |
| `--json` | machine-readable output instead of the text report |
| `--fast` | one OCR pass instead of two (~2× faster, slightly lower recall) |
| `--style embossed` | use the embossed column of the Rule 7 table |
| `--panel-area-cm2 350` | needed only when net quantity is by length/area/count |
| `--lang eng+hin` | if you installed the Hindi language pack |

A whole folder at once:

```bash
python backend/lmpc/cli.py batch ~/Desktop/products/ --marker-mm 39.4 --save
```

## Scanning with a webcam

```bash
python backend/lmpc/cli.py camera --marker-mm 39.4
```

A window opens with a live overlay: green PASS, amber BORDERLINE, red FAIL, plus the
current px/mm and tilt. Hold the pack and card in view, square-on.

- **space** — freeze the current reading, save it to history, print the full report
- **q** — quit

macOS will ask for camera permission the first time; grant it to the terminal app.

The assessment refreshes roughly **twice a second**, not every frame — the video is
smooth but the verdict lags it. Treat it as a live viewfinder that reassesses
periodically, and press space once the overlay is stable.

Other options: `--device 1` (external webcam), `--width 1920 --height 1080`,
`--interval 0.7` (seconds between assessments).

## Using the API instead

```bash
uvicorn lmpc.api:app --app-dir backend --port 8000

curl -X POST localhost:8000/api/scan \
  -F "file=@product.jpg" -F "marker_length_mm=39.4"
```

Interactive docs at <http://localhost:8000/docs>. Other endpoints: `/api/scans`
(history), `/api/stats` (dashboard totals), `/api/rules` (every legal threshold the
system applies), `/api/scans/{id}/report.txt`.

## Reading the result

Check the calibration block first — if that is wrong nothing below it means anything:

```
   method ArUco reference marker, marker id 0 (39.4 mm)
   scale  15.662 px/mm at marker | tilt 14.84 deg | taper 1.011
```

- **px/mm** — 16+ is good, 12–16 usable, under 12 means get closer
- **tilt** — under ~25° is fine

Then each declaration reports its measured height, the required minimum, the rule row
it came from, and one of:

- **PASS** — comfortably above the minimum
- **FAIL** — comfortably below it
- **BORDERLINE** — inside the measurement error band; a human should check with a loupe
- **NOT ASSESSED** — the system refused, and says why

**NOT ASSESSED is not a failure of the product, it is the tool declining.** Common
causes and fixes:

| message | fix |
|---|---|
| No ArUco reference marker found | the card is not in frame, is cut off, or is blown out by glare |
| imaged at only N px/mm | move closer, or use a higher-resolution camera |
| Measured N numeral(s) but the recogniser read M | glyphs merged or blurred — reshoot sharper |
| identified only by a value pattern | the key phrase ("MRP", "Net Qty.") was not read; the height is shown but not judged |
| not legible enough to claim the declaration is absent | the whole frame is too poor to say anything is missing |

## Before using results for anything real

The shipped error band is fitted on synthetic images. Fit it on **your** camera:

```bash
open backend/assets/height_calibration_sheet_A4_600dpi.pdf   # print at 100%
# photograph it 6-10 times, at different distances and angles, card in frame
python backend/lmpc/cli.py calibrate-physical ~/Desktop/calib_photos/ \
    --marker-mm 39.4
```

This rewrites `backend/data/uncertainty.json` with a band measured through your actual
lens, and the reports will then say "PHYSICAL calibration" instead of "SYNTHETIC".
This is also the evidence to show judges: it demonstrates the error band was measured,
not assumed.

## Known scope

Works: flat printed packaging, Latin script, printed (not embossed) declarations.

Does not work: curved surfaces (bottles, jars, cans), embossed or debossed text,
determining the common/generic name (reported as not machine-determinable, never as a
violation), and Rule 7 Table II without a supplied panel area.
