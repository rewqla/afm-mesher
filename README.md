# AFM Mesher

Проєкт для побудови 2D трикутної сітки методом **Advancing Front Method (AFM)** з:
- обробкою бінарних зображень (виділення зовнішнього контуру),
- генерацією сітки всередині контуру,
- оцінкою якості сітки,
- експортом у `OBJ`, `PLY`, `VTK`,
- інтерактивним UI-редактором маски.

## Що реалізує проєкт

Основний сценарій: взяти контур фігури (зображення або намальована маска) і отримати триангуляцію області.

Пайплайн:
1. Бінаризація зображення (`threshold`).
2. Виділення граничних ребер і зшивання в контури.
3. Спрощення контуру (Ramer-Douglas-Peucker, `epsilon`).
4. AFM-триангуляція (front-ребра, підбір кандидатів, перевірка валідності трикутників).
5. Пост-обробка (Laplacian smoothing).
6. Розрахунок метрик якості (min/mean/max + гістограма).
7. Експорт/візуалізація.

## Алгоритми

### Назви застосованих алгоритмів

- **Advancing Front Method (AFM)** — основна триангуляція області.
- **Laplacian Smoothing** — згладжування внутрішніх вузлів сітки.
- **Ramer-Douglas-Peucker (RDP)** — спрощення контуру (`epsilon`).
- **Ear Clipping Triangulation** — fallback-триангуляція при зупинці фронту.
- **Ray Casting Point-in-Polygon test** — перевірка належності точки полігону.
- **Segment Intersection Test (orientation-based)** — перевірка перетинів ребер.

### 1) Виділення межі з маски
- `src/infrastructure/image/photo_preprocessor.py`
- `src/infrastructure/processing/image_boundary_extractor.py`

Ключові ідеї:
- межа формується з ребер foreground-пікселів;
- ребра зшиваються в замкнені контури;
- для складних випадків обирається найбільший зовнішній контур за площею;
- контур спрощується RDP (`simplify_contour`).

### 2) Advancing Front Mesher
- `src/application/services/advancing_front_mesher.py`

Ключові кроки:
- нормалізація boundary (видалення дублікатів, CCW-орієнтація, перевірка площі);
- підподіл boundary до кроку `h` (`subdivide_boundary`);
- підтримка active front (список ребер);
- побудова кандидатів:
  - Steiner-кандидати (вздовж нормалі + зсуви),
  - підбір сусідніх існуючих вершин,
  - розширений радіус замикання фронту;
- геометричні перевірки:
  - орієнтація, належність полігону,
  - мінімальна якість трикутника,
  - відсутність перетинів з front,
  - відсутність включення сторонніх front-вершин;
- fallback: ear clipping, якщо AFM-фронт зупинився.

### 3) Покращення якості сітки

Якість покращується не одним прийомом, а комбінацією:
1. **Багатоспробна генерація**: до 4 спроб із зменшенням `h` (`h * 0.8^attempt`), вибір найкращої сітки за середньою якістю.
2. **Контроль мінімальної якості трикутників** (`min_triangle_quality`) ще під час генерації.
3. **Steiner-точки** для локального “вирівнювання” геометрії в місцях, де фронт інакше формує погані трикутники.
4. **Laplacian smoothing** лише внутрішніх вузлів, із фіксацією boundary.
5. **Перевірка орієнтації при згладжуванні** (не допускається переворот елементів).
6. **Зовнішній quality-gate у тестах**: для базових і складних форм очікується середня якість `>= 0.7`.

## Режими роботи

### 1) Batch режим
Команда:
```bash
python src/main.py --mode batch
```

Що робить:
- проганяє набір еталонних зображень із `data/images`,
- будує сітки,
- зберігає дебаг-візуалізації (`*_debug.png`),
- експортує `obj/ply/vtk`,
- генерує звіти:
  - `data/output/mesh_quality_report.json`,
  - `data/output/mesh_benchmark_report.csv`.

### 2) UI режим
Команда:
```bash
python src/main.py --mode ui
```

Що доступно:
- редактор маски (Pen, Eraser, Fill, Point, Segment, Rectangle, Circle),
- undo/redo, load/save,
- запуск триангуляції в окремому потоці.

Пресети триангуляції в UI (`TriangulationAdapter`):
- `Fast`
- `Balanced`
- `Accurate`
- `Custom` (ручні параметри: `h`, smoothing, contour epsilon, max iterations factor, min quality).

## Структура проєкту

```text
src/
  application/
    services/advancing_front_mesher.py
    dto/complex_test_contours.py
  domain/
    entities/ (Point, Edge, Triangle, Polygon, Mesh)
    geometry/geometry_utils.py
    interfaces/imesh_generator.py
  infrastructure/
    image/photo_preprocessor.py
    processing/image_boundary_extractor.py
    export/mesh_exporter.py
    visualization/debug_visualizer.py
  presentation/
    paint_app.py
    triangulation_adapter.py
    canvas.py, tools.py, main.py
scripts/
  generate_test_images.py
  run_stress_benchmarks.py
tests/
```

## Тести

Покриті основні класи ризиків:
- коректність доменних сутностей і геометричних утиліт;
- інваріанти AFM (позитивні площі, відсутність самоперетинів, належність полігону);
- стабільність на складних/випадкових/стресових кейсах;
- якість сітки та порогові критерії;
- інтеграційний pipeline “зображення -> boundary -> mesh -> debug output”;
- експорт у формати;
- preprocessing/contour extraction.

Основні тестові модулі:
- `test_afm_invariants.py`
- `test_afm_complex_quality.py`
- `test_afm_mesh_integrity_extra.py`
- `test_afm_randomized_regression.py`
- `test_afm_stress_cases_extra.py`
- `test_afm_subdivision_and_smoothing.py`
- `test_image_quality_thresholds.py`
- `test_main_pipeline.py`
- та інші в каталозі `tests/`.

Запуск:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Скрипти

- `scripts/generate_test_images.py` — генерація синтетичних тестових зображень.
- `scripts/run_stress_benchmarks.py` — серія запусків із різними `h` + метрики часу/пам’яті/якості.

## Залежності

`requirements.txt`:
- `PySide6==6.10.0`
- `Pillow==12.2.0`
- `matplotlib==3.10.7`

## Швидкий старт

```bash
pip install -r requirements.txt
python src/main.py --mode ui
```

або для пакетного прогону:

```bash
python src/main.py --mode batch
```
