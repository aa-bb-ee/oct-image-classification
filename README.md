# OCT Classification Project

## Projektziel
Automatische Klassifikation von OCT-Bildern in:
- CNV
- DME
- DRUSEN
- NORMAL

---

## Projektstruktur
```text 
oct_project/
│
├── data/                      		
│   └── OCT/
│       ├── train/             		
│       └── test/              		
│
├── experiment_outputs/        		# Output-Ordner für alle Experimente
│   └── [YYYYMMDD_HHMMSS_run_name]/
│       ├── models/            		
│       ├── reports/           		
│       └── runs/              		
│
├── src/
│   ├── __init__.py            	
│   ├── config.py              		# Definiert die PipelineConfig-Dataclass mit allen Standardwerten.
│   ├── gpu.py                 		# Übernimmt die automatische VRAM-Abfrage und die hardwareseitige GPU-Initialisierung.
│   ├── data_loader.py         		# Lädt die Bilder, baut die tf.data-Pipelines und berechnet die Kassengewichte.
│   ├── model.py               		# Definiert die InceptionV3-Architektur inkl. Augmentation, Head und Layer-Unfreezing.
│   ├── callbacks.py           		# Erstellt die Keras-Callback-Listen (EarlyStopping, Checkpoints) für die Trainingsphasen.
│   ├── training.py            		# Steuert den kompletten Trainingsprozess über Stage 1 (Feature Extraction) und Stage 2 (Fine-Tuning).
│   ├── evaluation.py          		# Lädt das beste Modell und berechnet Vorhersagen sowie erweiterte Metriken (ROC-AUC, F1).
│   ├── reporting.py           		# Generiert alle Plots (Lernkurven, Confusion Matrix) und exportiert die finalen JSON/CSV-Reports.
│   ├── paths.py			# Steuert zentral die Pfade der experiment_outputs.
│   └── helpers.py			# Hilfsfunktionen, die in keinem anderen Script sauber hineinpassen.
│
├── runs/				# Test-Scripte --> .gitignore
├── models/				# Annas Test-Ergebnisse --> .gitignore
|
├── cli/                       		
│   └── train.py			# Der zentrale Einstiegspunkt, der CLI-Argumente parst und den Ablauf in src/ orchestriert.
│
├── README.md                  		
└── requirements.txt           		
```

## Setup
source .venv/bin/activate

## JupyterLab starten
1. Auf dem lokalen Rechner den SSH-Tunnel öffnen:

   ```bash
   ssh -p 24 -i ~/.ssh/id_ed25519 -L 8888:localhost:8888 wfp_ai1@77.237.53.194
   ```
> **Hinweis:** Link zum private Key ggf. anpassen (~/.ssh/id_ed25519).
> **Hinweis:** Der Zugriff auf Jupyter erfolgt über einen SSH-Tunnel und setzt ein funktionierendes SSH-Keypair voraus. Evtl. auch mit RSA Key verbinden:

   ```bash
   ssh -p 24 -i ~/.ssh/id_rsa -L 8888:localhost:8888 wfp_ai1@77.237.53.194
   ```

2. Auf dem Server das virtuelle Environment (venv) aktivieren und JupyterLab starten:

   ```bash
   source ~/oct-project/.venv/bin/activate
   jupyter lab --no-browser --port=8888
   ```

3. Den im Terminal ausgegebenen Link inkl. Token im lokalen Browser öffnen:

   ```text
   http://localhost:8888/lab?token=...
   ```

>**Hinweis:** Der Token wird bei jedem Start von Jupyter neu erzeugt und im Terminal angezeigt.

---

## CLI Arguments

Die Trainingspipeline kann vollständig über Kommandozeilenargumente konfiguriert werden.

### Allgemeine Verwendung

```bash
python cli/train.py [ARGS]
```

---

### Verfügbare Argumente

| Argument | Typ | Default | Werte | Beschreibung |
|---|---:|---:|---|---|
| `--data_dir` | `str` | `data/OCT` | Pfad | Basisverzeichnis des Datensatzes |
| `--img_size` | `int` | `299` | `> 0` | Zielgröße der Bilder (quadratisch) |
| `--batch_size` | `int` | `32` | `> 0` | Batch Size |
| `--epochs` | `int` | `10` | `>= 0` | Anzahl Epochen Stage 1 (Feature Extraction) |
| `--fine_tune_epochs` | `int` | `10` | `>= 0` | Zusätzliche Epochen für Fine-Tuning |
| `--learning_rate` | `float` | `1e-4` | `> 0` | Lernrate Stage 1 |
| `--fine_tune_lr` | `float` | `1e-5` | `> 0` | Lernrate für Fine-Tuning |
| `--dropout` | `float` | `0.3` | `0.0 <= x < 1.0` | Dropout vor dem Classification Head |
| `--unfreeze_last_n` | `int` | `50` | `>= 0` | Anzahl freizugebender Backbone-Layer beim Fine-Tuning |
| `--run_name` | `str` | `oct_exp` | String | Name des Trainingslaufs |
| `--model_name` | `str` | `inceptionv3` | aktuell `inceptionv3` | Modellarchitektur / Backbone |
| `--gpu_index` | `int` | `-1` | `-1`, `0`, `1`, ... | GPU-Auswahl (`-1` = automatische Auswahl der GPU mit dem meisten freien Speicher) |
| `--train_take` | `int` | `-1` | `-1` oder `> 0` | Begrenzt Trainings-Batches (z. B. für Smoke Tests) |
| `--val_take` | `int` | `-1` | `-1` oder `> 0` | Begrenzt Validation-Batches |
| `--test_take` | `int` | `-1` | `-1` oder `> 0` | Begrenzt Test-Batches |
| `--seed` | `int` | `42` | Integer | Seed für Reproduzierbarkeit |
| `--val_split` | `float` | `0.1` | `0 < x < 1` | Anteil Validation Split vom Trainingsset |
| `--cache` | Flag | `False` | Flag | Aktiviert Dataset-Caching |
| `--mixed_precision` | Flag | `False` | Flag | Aktiviert Mixed Precision Training |
| `--fine_tune` | Flag | `False` | Flag | Aktiviert Fine-Tuning nach Stage 1 |
| `--use_class_weights` | Flag | `False` | Flag | Berechnet balancierte Klassengewichte |
| `--use-augmentation` | Bool Flag | `True` | Flag | Aktiviert Data Augmentation |
| `--no-use-augmentation` | Bool Flag | — | Flag | Deaktiviert Data Augmentation |

---

### Boolean Flags

#### Aktivieren

```bash
--fine_tune
--cache
--mixed_precision
--use_class_weights
```

Beispiel:

```bash
python cli/train.py --fine_tune --mixed_precision
```

---

#### Data Augmentation deaktivieren

Standardmäßig aktiv.

Deaktivieren mit:

```bash
python cli/train.py --no-use-augmentation
```
