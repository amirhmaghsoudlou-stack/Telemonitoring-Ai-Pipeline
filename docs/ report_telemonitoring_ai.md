## English

### 1) Task
We build an early-warning system: predict whether SpO2 will fall below 98 within the next 5 minutes using recent vital-sign history.

### 2) Data
UQ Vital Signs Dataset (trend signals per case). We work with time-indexed HR and SpO2 series. Raw data is not committed to GitHub.

### 3) Method
- Preprocessing: numeric conversion, time ordering, optional 1-second resampling for dense signals
- Labeling: future minimum SpO2 in a 5-minute horizon
- Features: 60-second rolling statistics + short-term SpO2 features (fast-drop oriented)
- Model: Gradient Boosting classifier
- Threshold: selected to target recall = 0.80 on the training set

### 4) Results (summary)
- Multi-case evaluation on a small case list (leave-one-case-out style)
- Metrics saved to `results/metrics/`
- Visualizations saved to `results/figures/` (examples: case01, case03, case04)

### 5) Discussion
- Trade-off between missed events (FN) and false alarms (FP)
- Case variability and occasional single-class cases
- Visual inspection shows that model alerts often appear before true drops in several cases, while fast/short drops remain more challenging

---

## Deutsch

### 1) Aufgabe
Wir bauen ein Frühwarnsystem: Vorhersage, ob SpO2 innerhalb der nächsten 5 Minuten unter 98 fällt, basierend auf der jüngsten Vitaldaten-Historie.

### 2) Daten
UQ Vital Signs Dataset (Trend-Signale pro Fall). Wir nutzen zeitindexierte Reihen von HR und SpO2. Rohdaten werden nicht in GitHub eingecheckt.

### 3) Methode
- Vorverarbeitung: numerische Konvertierung, zeitliche Sortierung, optionales 1-Sekunden-Resampling bei sehr dichter Abtastung
- Labeling: zukünftiges Minimum von SpO2 im 5-Minuten-Horizont
- Merkmale: Rolling-Statistiken (60 Sekunden) + Kurzzeit-Merkmale für schnelle Abfälle
- Modell: Gradient Boosting Klassifikator
- Schwelle: so gewählt, dass im Training Recall = 0.80 erreicht wird

### 4) Ergebnisse (Kurzfassung)
- Mehr-Fall-Auswertung auf einem kleinen Fall-Set (leave-one-case-out)
- Metriken unter `results/metrics/`
- Visualisierungen unter `results/figures/` (z.B. case01, case03, case04)

### 5) Diskussion
- Trade-off zwischen verpassten Ereignissen (FN) und Fehlalarmen (FP)
- Variabilität zwischen Fällen und gelegentliche Single-Class-Fälle
- Visuelle Analyse: Warnungen erscheinen in mehreren Fällen häufig vor den tatsächlichen Abfällen; schnelle/kurze Abfälle bleiben schwieriger