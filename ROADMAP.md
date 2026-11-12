# FilmDev Roadmap — Objektiv korrekte Farben

**Ziel:** Ein Algorithmus der Farb-Negative so invertiert wie ein kalibrierter Noritsu/Frontier-Scanner.
**Daten-Strategie:** Der Nutzer notiert bei jedem Scan welchen Filmstock er benutzt hat → wir bauen organisch eine Datenbank auf → Stufe 3 trainierbar.

---

## Was Noritsu/Frontier/Pakon wirklich machen

Industriescanner haben für jeden Filmstock gespeichert:
- **Masken-Farbe** (wie orange ist dieser Film genau, R/G/B-Verhältnis)
- **Sensitometry-Kurven** (wie reagieren die drei Dye-Schichten auf Licht)
- **Density-Range** (was ist typisches Schwarz, typisches Weiß für diesen Stock)
- **Farbbalance-Ziel** (wohin soll Hauttoneutralität zielen)

Kein Machine Learning — reine Colorimetrie-Tabellen, jahrzehntelang kalibriert.
Das können wir nachbauen. Vollständig.

---

## Stufe 1 — "Technisch korrekt" (heute ✅ fast fertig)

**Was wir haben:**
- WB von Film Base Maske (G-Referenz, kein Mean)
- `1/x` Transmittance Inversion (mathematisch korrekt)
- Auto-Levels Percentile

**Was fehlt noch:**
- Ergebnis ist technisch korrekt aber **flach** (kein Kontrast, kein Filmlook)
- Genau das was Noritsu als "Rohscan vor Lab-Kurve" ausgibt
- Das ist der **gewollte Ausgangspunkt** für Color Grading

**Wann fertig:** Morgen nach erstem echten Test → ggf. kleine Tweaks

---

## Stufe 2 — Filmstock-Profile (nächstes Wochenende)

### Konzept
User wählt Filmstock aus Dropdown → wir laden eine gespeicherte Kurve:
- **Masken-Referenzfarbe** für diesen Stock (statt User-Klick als einzige Quelle)
- **Tonwertkurve** die zum Stock passt (Kontrast, Gamma)
- **Farbbalance-Offset** (Kodak ist warm, Fuji ist kühler/grüner)
- **Sättigung** (Velvia: hoch, Portra: medium)

### Quellen für Kurven/Profile (recherchiert 2026-07-15)

**Aaron Buchler LUTs (Open Source, MIT):**
- Repo: https://github.com/abpy/color-neg-resources
- Enthält: `.cube` LUTs für Inversion, Density-to-Luminance, Paper Print Simulation
- Direkt verwendbar — in Python via `colour` oder `pillow-lut`

**YAFIS — Yet Another Film Inversion Software:**
- Repo: https://github.com/jad3675/YAFIS
- Hat: automatische Filmstock-Erkennung (C-41 / ECN-2 / E-6 / B&W)
- Source lesen → Algorithmus abschauen für Auto-Detect

**NegICC — ICC Profile für Farbnegative:**
- Repo: https://github.com/arufahc/negicc
- ICC Profiles für rawpy direkt verwendbar (`output_color=rawpy.ColorSpace.raw` + ICC)

**Alchemy Color Method:**
- Repo: https://github.com/alchemy-color/color-negative-inversion
- Nutzt reversed Adobe DNG Standard Curve → besser als plain `1/x` für finale Looks

### Filmstock-Liste (Startset)

| Filmstock | Typ | Masken-Charakteristik | Priorität |
|-----------|-----|----------------------|-----------|
| Kodak Gold 200 | C-41 | Warm orange, niedrige Sättigung | ⭐ hoch (Einsteiger-Film) |
| Kodak Portra 400 | C-41 | Warm orange, skin-tone-optimiert | ⭐ hoch |
| Kodak Portra 160 | C-41 | Wie 400, etwas kühler | mittel |
| Kodak ColorPlus 200 | C-41 | Kräftig orange, gesättigt | mittel |
| Kodak Ultramax 400 | C-41 | Ähnlich Gold, etwas grüner | mittel |
| Fujifilm Superia 400 | C-41 | Kühler, grünlicher Stich | ⭐ hoch |
| Fujifilm 400H Pro | C-41 | Sehr kühl, pastell | mittel |
| Fujifilm Velvia 50 | E-6 (Slide) | Kein Masken-Problem, direkt invertieren | niedrig |
| Kodak Ektar 100 | C-41 | Sehr gesättigt, kühl | mittel |
| Ilford HP5 Plus | B&W | Keine Maske — einfache Inversion | ⭐ hoch |
| Kodak T-Max 400 | B&W | Keine Maske | mittel |
| Fujifilm Neopan 400 | B&W | Keine Maske | niedrig |

### Implementation Plan Stufe 2

```python
# core/filmstock_profiles.py
PROFILES = {
    "kodak_gold_200": {
        "mask_ref": [0.82, 0.52, 0.18],   # typische Masken-RGB (float32)
        "gamma": 1.15,                      # leicht erhöhter Kontrast
        "saturation": 1.05,                 # leicht erhöht
        "wb_offset": [1.02, 1.0, 0.97],   # warm
        "curve": "gentle_s",               # S-Kurve Typ
    },
    "fuji_superia_400": {
        "mask_ref": [0.75, 0.55, 0.25],   # kühler/grünlicher
        "gamma": 1.1,
        "saturation": 1.08,
        "wb_offset": [0.98, 1.0, 1.03],   # leicht kühl
        "curve": "gentle_s",
    },
    # ... etc
}
```

---

## Stufe 3 — Auto-Filmstock-Erkennung (2-3 Wochen, braucht Daten)

### Wie es funktioniert

Die **Farbe der Filmmaske** ist filmstock-spezifisch:
- Kodak (C-41): R/G/B Verhältnis ~4:2.5:1 (warm orange)
- Fuji (C-41): R/G/B Verhältnis ~3.5:2.5:1.2 (etwas kühler, grünlicher)
- B&W: R≈G≈B (neutral grau)
- Slide (E-6): fast keine Maske

Wir können aus dem Masken-Sample alleine schon ableiten ob es Kodak oder Fuji ist — ohne EXIF.

### Daten-Strategie (Workflow)

**Aufgabe des Nutzers bei jedem Scan:**
1. Filmstock notieren (→ Dropdown in FilmDev auswählen)
2. Maske klicken wie immer

**Was wir speichern pro Scan:**
```json
{
  "filmstock": "kodak_gold_200",
  "mask_sample": [0.81, 0.51, 0.19],
  "mask_sample_stddev": [0.02, 0.01, 0.01],
  "camera": "Sony A7RII",
  "lens": "...",
  "backlight": "...",
  "date": "2026-07-16"
}
```

**Nach ~50 Scans verschiedener Stocks:**
- Clustering (k-Means oder einfaches Nearest-Neighbor) → automatische Erkennung
- kein Deep Learning nötig — Farbraum ist klein genug für klassisches ML
- Genauigkeit nach 50 Samples: schätzungsweise 85-90% für häufige Stocks

### Implementation Plan Stufe 3

```python
# core/filmstock_detector.py
class FilmstockDetector:
    def detect(self, mask_sample: np.ndarray) -> str:
        """Gibt wahrscheinlichsten Filmstock zurück."""
        # 1. B&W check: wenn R≈G≈B → kein Farbnegativ
        # 2. Slide check: wenn alle Werte >0.7 → E-6
        # 3. Nearest neighbor in Masken-Farbraum → Filmstock
        ...
    
    def add_training_sample(self, mask_sample, filmstock: str):
        """Speichert neuen Sample in data/filmstock_samples.json"""
        ...
```

---

## Datenbank-Format (ab Stufe 2)

**Wo:** `data/scan_log.json`
**Format:**

```json
[
  {
    "date": "2026-07-16",
    "file": "DSC00123.ARW",
    "filmstock": "kodak_gold_200",
    "mask_rgb": [0.812, 0.513, 0.187],
    "wb_mults": [0.632, 1.0, 2.746],
    "notes": "Fujifilm X-S10 Backlight, ISO 100"
  }
]
```

---

## LUT-Integration (Stufe 2)

Aaron Buchler stellt `.cube` LUTs bereit die wir direkt einlesen können:

```python
# core/lut_engine.py
import numpy as np

def load_cube_lut(path: str) -> dict:
    """Lädt eine .cube 3D-LUT."""
    ...

def apply_lut(image: np.ndarray, lut: dict) -> np.ndarray:
    """Interpoliert image durch 3D-LUT."""
    ...
```

**Verfügbare LUTs (von abpy/color-neg-resources):**
- `inverse.cube` — skalierte 1/x Inversion (besser kalibriert als unser plain 1/x)
- `transmittance_to_density.cube` — für density-basierte Workflows
- `paper_print.cube` — simuliert optischen Fotoabzug auf Baryt-Papier (warmer Look)

---

## Scan-Workflow

Um die Datenbank aufzubauen bitte bei jedem Scan:

1. **Filmstock notieren** — entweder auf der Kassette/Verpackung oder Gedächtnis
2. **In FilmDev:** Filmstock-Dropdown wählen (kommt in Stufe 2)
3. **Maske klicken** wie gewohnt
4. Optional: kurze Notiz (Backlight-Typ, Beleuchtung wenn anders)

Das wars. Nach 20-30 Scans pro Stock haben wir genug für Stufe 3.

**Empfohlene Dokumentation pro Filmrolle:**
```
Film: Kodak Gold 200
Datum: 2026-07-16
Kamera: Sony A7RII
Belichtung: ISO 200, Tageslicht
Backlight: iPad (Standard)
Besonderheiten: -
```

---

## Timeline

| Wann | Was |
|------|-----|
| **Morgen (2026-07-16)** | Erster echter Test, Farbfeedback, ggf. Levels-Tweaks |
| **Nächstes WE** | Stufe 2: Filmstock-Dropdown + erste Profile (Gold 200, Superia 400, HP5) |
| **Nach 20+ Scans** | Masken-Datenbank auswerten, Cluster prüfen |
| **Nach 50+ Scans** | Stufe 3: Auto-Detect implementieren |
| **Langfristig** | EXE bauen, ggf. als Tool veröffentlichen |

---

## Quellen & Referenzen

- **Aaron Buchler Guide** (Goldstandard): https://abpy.github.io/2023/08/20/color-neg.html
- **Aaron Buchler LUTs + Scripts**: https://github.com/abpy/color-neg-resources
- **YAFIS (Auto-Detect Referenz)**: https://github.com/jad3675/YAFIS
- **NegICC (ICC Profile Ansatz)**: https://github.com/arufahc/negicc
- **Alchemy Color Method**: https://github.com/alchemy-color/color-negative-inversion
- **Negative Viewer Guide**: https://negativeviewer.tokugai.com/guides/orange-mask-removal
