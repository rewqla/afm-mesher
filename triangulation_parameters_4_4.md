# Параметри тріангуляції в застосунку

Фактичний опис реалізації для підготовки розділу 4.4. Посилання наведено у форматі `файл:рядок`; де причина вибору числової константи не задокументована, це зазначено явно.

## 1. Параметри, що задає користувач

- У правій панелі є пресет тріангуляції, прапорець користувацьких значень, цільова довжина ребра `h`, ітерації згладжування, епсілон спрощення контуру, коефіцієнт максимальної кількості ітерацій, мінімальна якість трикутника та масштаб у метрах на піксель. — `src/presentation/paint_app.py:541-617`
- Користувач також може змінювати розмір полотна, тему й мову. — `src/presentation/paint_app.py:479-523`

## 2. Елементи UI, що впливають на тріангуляцію

Параметри розміщені у правій панелі Properties; окремих діалогів редагування параметрів немає. Інформаційний діалог описує включення та розрізи, але не змінює параметрів. — `src/presentation/paint_app.py:466-471`, `src/presentation/paint_app.py:1214-1227`

- **Preset** — `QComboBox`: Fast, Balanced, Accurate, Custom; початкове значення Balanced. — `src/presentation/paint_app.py:231`, `src/presentation/paint_app.py:541-547`, `src/presentation/paint_app.py:620`
- **Use custom values** — `QCheckBox`, початково вимкнений; вмикає редагування полів пресета. — `src/presentation/paint_app.py:549-551`, `src/presentation/paint_app.py:1178-1198`
- **Target edge length (h)** — `QDoubleSpinBox`; діапазон 4,0–80,0 координатних одиниць, крок 1,0, точність 1 десятковий знак. — `src/presentation/paint_app.py:558-566`, `src/presentation/triangulation_adapter.py:45-46`, `src/presentation/triangulation_adapter.py:375-379`
- **Smoothing iterations** — `QSpinBox`; 0–10 ітерацій, крок 1. — `src/presentation/paint_app.py:568-575`, `src/presentation/triangulation_adapter.py:47-48`, `src/presentation/triangulation_adapter.py:380-383`
- **Contour simplify epsilon** — `QDoubleSpinBox`; 0,5–10,0 координатних одиниць, крок 0,1, точність 2 десяткові знаки. — `src/presentation/paint_app.py:577-585`, `src/presentation/triangulation_adapter.py:49-50`, `src/presentation/triangulation_adapter.py:384-386`
- **Max iterations factor** — `QSpinBox`; 50–1000, крок 10. Це множник; ліміт циклу обчислюється окремо. — `src/presentation/paint_app.py:587-594`, `src/presentation/triangulation_adapter.py:51-52`, `src/presentation/triangulation_adapter.py:388-392`
- **Min triangle quality** — `QDoubleSpinBox`; 0,0–1,0, крок 0,01, точність 3 десяткові знаки. — `src/presentation/paint_app.py:596-604`, `src/presentation/triangulation_adapter.py:53-54`, `src/presentation/triangulation_adapter.py:393-396`
- **Scale (m/px)** — `QDoubleSpinBox`; 10⁻⁶–10⁶ м/піксель, крок 0,001, точність 6 десяткових знаків, початково 1,0 м/піксель. — `src/presentation/paint_app.py:606-617`, `src/presentation/triangulation_adapter.py:55-56`
- Поля мають обмеження Qt spin box за діапазоном; адаптер також перевіряє отримані custom values і піднімає `ValueError` за виходу за межі. — `src/presentation/paint_app.py:558-617`, `src/presentation/triangulation_adapter.py:375-400`

## 3. Пресети та ручне редагування

- **Fast:** `h=28,0`; згладжування 2 ітерації; epsilon 3,0; множник ітерацій 180; мінімальна якість 0,01; масштаб 1,0 м/піксель. — `src/presentation/triangulation_adapter.py:748-757`
- **Balanced:** `h=20,0`; згладжування 5 ітерацій; epsilon 2,0; множник 260; мінімальна якість 0,01; масштаб 1,0 м/піксель. — `src/presentation/triangulation_adapter.py:758-766`
- **Accurate:** `h=14,0`; згладжування 8 ітерацій; epsilon 1,0; множник 400; мінімальна якість 0,01; масштаб 1,0 м/піксель. — `src/presentation/triangulation_adapter.py:767-775`
- Вибір пресета записує його значення у поля. Поки прапорець custom не ввімкнений, поля вимкнені; після ввімкнення внутрішній режим стає Custom, хоча вибраний пункт комбобокса лишається попереднім. — `src/presentation/paint_app.py:1155-1159`, `src/presentation/paint_app.py:1168-1176`, `src/presentation/paint_app.py:1178-1198`
- Ручне редагування не перемикає комбобокс пресета; значення залишаються в полях. Повторний вибір пресета знову застосовує його значення. — `src/presentation/paint_app.py:546-551`, `src/presentation/paint_app.py:1155-1176`

## 4. `target_h`: зміст, одиниці та застосування

- У UI параметр має ім’я `target_edge_length`; у ядрі його передають як `target_h` або `target_step`. Це цільовий просторовий крок, з яким порівнюються довжини ребер; окремої одиниці у декларації налаштувань немає. — `src/presentation/triangulation_adapter.py:33-40`, `src/presentation/triangulation_adapter.py:105-110`, `src/application/services/advancing_front_mesher.py:148-157`
- Для контурів полотна координати задані у координатах полотна, тобто в пікселях; у цьому випадку `h` має ті самі координатні одиниці. — `src/presentation/canvas.py:21-27`, `src/presentation/canvas.py:294-295`
- `h` використовується при оптимізації оболонки, отворів і розрізів; допуск спрощення при децимації дорівнює `max(0,5; 0,1·h)`. — `src/presentation/triangulation_adapter.py:132-143`, `src/presentation/triangulation_adapter.py:811-833`
- Ядро використовує `h` для поділу оболонки/отворів/розрізів і для активації штейнерівських кандидатів та фільтра відстані до фронту. — `src/application/services/advancing_front_mesher.py:155-157`, `src/application/services/advancing_front_mesher.py:608-617`
- У пошуку кандидатів є near-радіус в одну довжину ребра та дальній радіус 2,5 довжини ребра. — `src/application/services/advancing_front_mesher.py:632-656`
- `meters_per_pixel` не масштабує координати для пошуку чи поділу ребер; фізична площа обчислюється множенням площі в координатах на квадрат масштабу. — `src/presentation/triangulation_adapter.py:186-196`, `src/domain/entities/linear_triangle.py:58-64`
- Масштабування `h` від розміру полотна в UI відсутнє; окремо працює адаптація за площею та оцінкою довжини фронту (п. 9). — `src/presentation/paint_app.py:558-566`, `src/presentation/triangulation_adapter.py:778-809`

## 5. Обмеження кількості точок

- Єдиного користувацького глобального ліміту точок немає; задані окремі межі: оболонка — 1400 точок, кожен отвір — 600, кожен розріз — 500. — `src/presentation/triangulation_adapter.py:132-140`
- При перевищенні ліміту точки децимуються з кроком `ceil(кількість / ліміт)`, після чого близькі до колінеарних точки спрощуються. — `src/presentation/triangulation_adapter.py:811-833`
- Перевищення цих лімітів саме по собі не показує помилку; адаптер спрощує вхідну геометрію. — `src/presentation/triangulation_adapter.py:811-833`
- Ядро додатково піднімає `h` до `max(perimeter / 240; 0,001)` координатних одиниць для бюджету початкового фронту. Це автоматичне коригування, не помилка. — `src/application/services/advancing_front_mesher.py:432-444`

## 6. Інші доступні користувачу параметри

- **Поріг якості:** поле Min triangle quality; передається через `TriangulationSettings` у `AdvancingFrontMesher`. — `src/presentation/paint_app.py:596-604`, `src/presentation/paint_app.py:1200-1209`, `src/presentation/triangulation_adapter.py:145-150`
- **Ітерації згладжування:** поле Smoothing iterations; передається в конструктор ядра. — `src/presentation/paint_app.py:568-575`, `src/presentation/triangulation_adapter.py:145-150`
- **Режим перенумерування вузлів** в UI відсутній; адаптер має стратегію за замовчуванням `rcm_multistart`. — `src/presentation/paint_app.py:466-622`, `src/presentation/triangulation_adapter.py:58-63`
- **Розмір полотна:** поля Width/Height і пресет розміру; застосовується через `Canvas.resize_canvas`. Розмір не є окремим полем `TriangulationSettings`; ядро отримує геометрію контурів. — `src/presentation/paint_app.py:493-523`, `src/presentation/paint_app.py:751-771`, `src/presentation/paint_app.py:274-289`
- **Масштаб пікселя до метра:** поле Scale (m/px), передається у Mesh. — `src/presentation/paint_app.py:606-617`, `src/presentation/paint_app.py:1200-1209`, `src/presentation/triangulation_adapter.py:186-196`
- Розмір полотна за замовчуванням — 900×650 px; діапазон — 256–4096 px на кожну вісь; крок полів — 16 px. Пресети: 640×480, 800×600, 1024×768, 1280×720, 1920×1080 px. — `src/presentation/canvas.py:16-27`, `src/presentation/paint_app.py:208-214`, `src/presentation/paint_app.py:509-523`

## 7. Збереження налаштувань

- Збереження параметрів між запусками через QSettings чи файл конфігурації відсутнє; початкові тема/мова можуть читатися зі змінних середовища `AFM_THEME` та `AFM_LANG`. — `src/presentation/paint_app.py:307-311`, `src/presentation/paint_app.py:479-484`
- Очищення полотна видаляє вміст і overlay сітки, але не змінює значення контролів параметрів. — `src/presentation/canvas.py:62-74`

## 8. Внутрішні параметри, недоступні користувачу

- Поріг достатньої середньої якості для раннього завершення — 0,7. — `src/application/services/advancing_front_mesher.py:101-105`, `src/application/services/advancing_front_mesher.py:127-133`
- `alpha` згладжування за замовчуванням дорівнює 0,5; для переміщення кожного вузла передбачено до 4 спроб із діленням `alpha` навпіл після невдалої спроби. — `src/application/services/mesh_postprocessing.py:373-385`, `src/application/services/mesh_postprocessing.py:416-438`
- Ліміт ітерацій одного проходу — `max(200, len(front) * max_iterations_factor)`. — `src/application/services/advancing_front_mesher.py:166-178`
- Межі кандидатного пошуку: максимум 256 ребер сканування; для деяких областей — `min(len(front), max(256, len(front)//3))`; 18 штейнерівських кандидатів на проході; активація Steiner при `edge_len > 0,95·target_step`; відсікання за близькістю — `0,4·target_step`. — `src/application/services/advancing_front_mesher.py:35-37`, `src/application/services/advancing_front_mesher.py:241-242`, `src/application/services/advancing_front_mesher.py:270-278`, `src/application/services/advancing_front_mesher.py:608-617`
- Near-радіус: `edge_len + 10⁻⁹`; far-радіус: `2,5·edge_len`; обмеження відношення сторін кандидата — 1,8 (near) і 2,2 (far); коефіцієнт максимальної діагоналі ear clipping — 2,4. — `src/application/services/advancing_front_mesher.py:33-35`, `src/application/services/advancing_front_mesher.py:265-268`, `src/application/services/advancing_front_mesher.py:632-656`, `src/application/services/advancing_front_mesher.py:1162-1174`
- RCM multistart: типовий ліміт 20 кандидатів, крок зондування 12, початковий пул до 15, до 3 кандидатів за ексцентриситетом; точна кількість запусків залежить від структури графа й дедуплікації. — `src/application/services/mesh_postprocessing.py:89-115`, `src/application/services/mesh_postprocessing.py:985-994`
- Причина вибору числових значень внутрішніх констант не задокументована. Для адаптивної гілки задокументовано лише загальну мету: зменшити вартість побудови великих областей та обсяг згладжування. — `src/application/services/advancing_front_mesher.py:33-37`, `src/application/services/mesh_postprocessing.py:373-380`, `src/presentation/triangulation_adapter.py:783-784`

## 9. Адаптивні налаштування

- Адаптер обчислює площу оболонки, її периметр і `rough_front_nodes = perimeter / max(target_h, 10⁻⁶)`. — `src/presentation/triangulation_adapter.py:785-792`
- Адаптація пропускається лише коли одночасно `area < 200 000` координатних одиниць² та `rough_front_nodes < 240`; інакше застосовується адаптивна гілка. — `src/presentation/triangulation_adapter.py:794-796`
- Адаптивна гілка задає `h = max(h, min(1,5·h, h+14))`, згладжування `min(початкове, 2)`, множник ітерацій `min(початковий, 180)`. — `src/presentation/triangulation_adapter.py:797-809`
- До цієї перевірки діють ліміти вершин оболонки/отворів/розрізів; у ядрі також коригується `h` за периметром до `max(perimeter/240, 0,001)`. — `src/presentation/triangulation_adapter.py:132-143`, `src/application/services/advancing_front_mesher.py:432-444`
- Адаптація саме за шумом контуру або числом включень відсутня; умова адаптера використовує площу оболонки й оцінку фронту. Наявність отворів змінює ліміт сканування кандидатів у ядрі. — `src/presentation/triangulation_adapter.py:778-809`, `src/application/services/advancing_front_mesher.py:539-543`

## 10. Послідовні спроби з різними `target_h`

- Коефіцієнти до базового `h`: 1,0; 1,2; 0,85; 0,7. — `src/application/services/advancing_front_mesher.py:103-108`
- Найкращою з успішних спроб вважається сітка з найбільшою середньою якістю; при середній якості ≥0,7 функція повертає результат і припиняє наступні спроби. — `src/application/services/advancing_front_mesher.py:101-105`, `src/application/services/advancing_front_mesher.py:127-146`
- Середня якість рахується за всіма трикутниками Mesh; окремого фільтру лише внутрішніх елементів у цій метриці немає. — `src/application/services/advancing_front_mesher.py:1040-1042`
- Послідовність та поріг 0,7 однакові для всіх режимів; пресет задає базові значення, а адаптивна гілка може змінити базовий `h` перед викликом mesher. — `src/application/services/advancing_front_mesher.py:103-108`, `src/presentation/triangulation_adapter.py:748-809`

## 11. Резервні сценарії та невдача

- Ear clipping викликається, якщо фронт одного проходу застряг і з нього вдалося скласти замкнену петлю; ще один резервний запуск виконується для простої області, коли немає отворів/розрізів і не вдалася жодна спроба AFM. — `src/application/services/advancing_front_mesher.py:187-199`, `src/application/services/advancing_front_mesher.py:135-145`, `src/application/services/advancing_front_mesher.py:1044-1054`
- Для розрізів послідовність така: початковий список; список після pruning колінеарних перекриттів; за наявності розрізів — повтор без розрізів. Наступна спроба запускається після `ValueError`. — `src/presentation/triangulation_adapter.py:159-181`
- UI-повідомлення про використання резервного сценарію відсутнє; перебіг спроб журналюється через `tri_debug`. — `src/presentation/triangulation_adapter.py:165-181`
- Якщо ліміт ітерацій вичерпано з непорожнім фронтом, ядро піднімає `ValueError("AFM failed to close the front before iteration limit.")`; UI показує текст винятку у попередженні. — `src/application/services/advancing_front_mesher.py:209-217`, `src/presentation/paint_app.py:1669-1673`
- Ліміту часу основного циклу немає; цикл обмежений числом ітерацій. — `src/application/services/advancing_front_mesher.py:166-178`

## 12. Повідомлення про невдалу тріангуляцію

- Заголовок вікна попередження — `Triangulation Error`, текст — рядок винятку; у статус-барі показується локалізований статус «Тріангуляція не вдалась» / «Triangulation failed». — `src/presentation/paint_app.py:73-75`, `src/presentation/paint_app.py:118-120`, `src/presentation/paint_app.py:1669-1673`
- Окремого ключа локалізації для причин помилки немає: у QMessageBox передається `str(error)`. — `src/presentation/paint_app.py:201-203`, `src/presentation/paint_app.py:1669-1672`

## 13. Візуалізація результату

- Сітка малюється ребрами трикутників суцільною червоною лінією товщиною 1 px; заливки та окремого малювання вузлів немає. Колір не залежить від якості трикутника чи типу вузла. — `src/presentation/canvas.py:517-527`
- Порядок на екрані: зображення, ескіз фігури, геометричні контури, маркери помилок, сітка. — `src/presentation/canvas.py:390-403`
- Зображення є базовим шаром. Перемикачів видимості зображення, сітки чи контурів у UI немає; сітка малюється, коли задано `mesh_overlay`. — `src/presentation/canvas.py:390-403`, `src/presentation/canvas.py:219-221`, `src/presentation/paint_app.py:466-622`
- Геометричний overlay: перший замкнений контур — `#2563eb`, суцільна лінія 2 px; наступні замкнені — `#16a34a`, пунктир 2 px; відкриті — `#dc2626`, суцільна 1 px. — `src/presentation/canvas.py:543-557`
- Масштабування коліщатком і переміщення полотна після побудови сітки в реалізації Canvas відсутні; події миші обробляють малювання. — `src/presentation/canvas.py:290-355`

## 14. Метрики у статус-барі

- Формат повідомлення: `Тріангуляцію завершено. коефіцієнт={coefficient:.4f}; вузлів={node_count}; макс. різниця вузлів={node_difference}`; англійський формат використовує `coefficient`, `nodes`, `max node difference`. — `src/presentation/paint_app.py:101-104`, `src/presentation/paint_app.py:147-150`, `src/presentation/paint_app.py:291-302`
- `coefficient` — середнє значення якості трикутників; якість одного трикутника `4·√3·area / (ab²+bc²+ca²)`, обмежена інтервалом [0,1]. — `src/domain/geometry/geometry_utils.py:90-108`, `src/domain/geometry/geometry_utils.py:134-137`
- `nodes` — число унікальних вузлів; `max node difference` — bandwidth/maxDifference індексованої сітки або обчислений розкид номерів вузлів. — `src/presentation/paint_app.py:1708-1722`, `src/application/services/mesh_postprocessing.py:21-23`
- Статус-бар також має постійні підписи інструмента, режиму, стану та індикатор прогресу без відсотка. — `src/presentation/paint_app.py:639-650`

## 15. Збережене зображення з накладеною сіткою

- Файл із сіткою формується через `Canvas.rendered_image()` і використовує той самий `_draw_mesh_overlay`, що й екран. — `src/presentation/paint_app.py:1252-1259`, `src/presentation/canvas.py:95-107`, `src/presentation/canvas.py:390-403`
- Розмір та кольори збігаються з буфером полотна; окремо записується оригінальне зображення без сітки. — `src/presentation/paint_app.py:1244-1259`, `src/presentation/canvas.py:95-107`
- Екран може показувати тимчасовий ескіз фігури, а `rendered_image()` його не додає. — `src/presentation/canvas.py:95-107`, `src/presentation/canvas.py:390-403`

## 16. Вплив теми й мови

- Тема змінює таблицю стилів UI; колір сітки задається Canvas незалежно від теми. Полотно початково біле. — `src/presentation/paint_app.py:652-680`, `src/presentation/canvas.py:26-27`, `src/presentation/canvas.py:517-527`
- Зміна мови оновлює заголовки, підписи, пресети та статусні рядки; код малювання сітки від мови не залежить. — `src/presentation/paint_app.py:313-370`, `src/presentation/canvas.py:517-527`

## 17. Зміна параметрів після побудови

- Зміна `h` або пресета сама по собі не запускає побудову й не очищає наявний `mesh_overlay`; нові параметри застосуються при наступній побудові. — `src/presentation/paint_app.py:1155-1176`, `src/presentation/paint_app.py:1200-1212`, `src/presentation/canvas.py:219-221`
- Початок малювання очищає наявну сітку; команда Clear також скидає overlay. — `src/presentation/canvas.py:62-74`, `src/presentation/canvas.py:298-300`

## 18. Зіставлення з твердженнями розділів 2–3

- **2.1 — відповідає:** є кілька спроб із різними `h`, відбір за середньою якістю, раннє завершення при порозі 0,7. Коефіцієнти: 1,0; 1,2; 0,85; 0,7. — `src/application/services/advancing_front_mesher.py:101-108`, `src/application/services/advancing_front_mesher.py:127-146`
- **2.4 — відповідає з уточненням:** `alpha=0,5`; до чотирьох спроб зі зменшенням удвічі — це спроби переміщення окремого вузла, не повтори всієї тріангуляції. — `src/application/services/mesh_postprocessing.py:373-385`, `src/application/services/mesh_postprocessing.py:416-438`
- **3.3 — частково відповідає:** є адаптація за площею/оцінкою фронту та децимація за кількістю вершин; окремі умови за шумом чи кількістю включень відсутні. — `src/presentation/triangulation_adapter.py:778-809`, `src/presentation/triangulation_adapter.py:811-833`
- **3.2 — не відповідає повністю:** Canvas має preprocess epsilon 0,75 і closure tolerance 8,0 координатних одиниць; адаптер перевіряє замикання за `max(1,0; 2·epsilon)`, а epsilon спрощення задається пресетом або користувачем. — `src/presentation/canvas.py:21-25`, `src/presentation/triangulation_adapter.py:534-544`, `src/presentation/triangulation_adapter.py:748-775`
- Округлення не має єдиної точності: для ключів отворів застосовано `round(..., 3)`, для ключів вузлів — 10 десяткових знаків. Причина точностей не задокументована. — `src/presentation/triangulation_adapter.py:676-685`, `src/application/services/advancing_front_mesher.py:33-38`, `src/application/services/advancing_front_mesher.py:1008-1020`

## 19. Джерело значень параметрів

| Параметр | Джерело значення | Файл:рядок |
|---|---|---|
| Preset Fast/Balanced/Accurate/Custom | [UI] Вибір у комбобоксі; поля синхронізуються з трьома готовими пресетами | `src/presentation/paint_app.py:541-547`, `src/presentation/paint_app.py:1168-1176` |
| `target_edge_length` / `target_h` | [UI] Пресет або поле; [АВТО] можливе збільшення адаптивною гілкою | `src/presentation/triangulation_adapter.py:748-775`, `src/presentation/triangulation_adapter.py:778-809` |
| `smoothing_iterations` | [UI] Пресет або поле; [АВТО] обмеження до 2 для великих областей | `src/presentation/paint_app.py:568-575`, `src/presentation/triangulation_adapter.py:801-808` |
| `contour_epsilon` | [UI] Пресет або користувацьке поле | `src/presentation/paint_app.py:577-585`, `src/presentation/triangulation_adapter.py:748-775` |
| `max_iterations_factor` | [UI] Пресет або поле; [АВТО] обмеження до 180 | `src/presentation/paint_app.py:587-594`, `src/presentation/triangulation_adapter.py:801-808` |
| `min_triangle_quality` | [UI] Пресет або користувацьке поле | `src/presentation/paint_app.py:596-604`, `src/presentation/triangulation_adapter.py:748-775` |
| `meters_per_pixel` | [UI] Поле; початково 1,0 | `src/presentation/paint_app.py:606-617` |
| Ширина/висота полотна | [UI] Spin box або пресет розміру | `src/presentation/paint_app.py:493-523` |
| Ліміти точок оболонки/отворів/розрізів | [КОНСТАНТА] 1400/600/500 точок; причина не задокументована | `src/presentation/triangulation_adapter.py:132-140` |
| Ліміт фронту | [КОНСТАНТА] 240 у формулі мінімального `h`; причина не задокументована | `src/application/services/advancing_front_mesher.py:35-37`, `src/application/services/advancing_front_mesher.py:432-444` |
| Адаптивні пороги площі й оцінки фронту | [КОНСТАНТА] 200 000 одиниць² і 240; застосування [АВТО]; причини точних значень не задокументовані | `src/presentation/triangulation_adapter.py:785-809` |
| Поріг раннього завершення за якістю | [КОНСТАНТА] 0,7; причина не задокументована | `src/application/services/advancing_front_mesher.py:101-105`, `src/application/services/advancing_front_mesher.py:127-133` |
| Коефіцієнти спроб `h` | [КОНСТАНТА] 1,0; 1,2; 0,85; 0,7; причина не задокументована | `src/application/services/advancing_front_mesher.py:103-108` |
| `alpha` згладжування | [КОНСТАНТА] 0,5; причина не задокументована | `src/application/services/mesh_postprocessing.py:373-380` |
| Кількість спроб згладжування вузла | [КОНСТАНТА] 4; причина не задокументована | `src/application/services/mesh_postprocessing.py:416-438` |
| Пороги Steiner та радіуси кандидатів | [КОНСТАНТА] 0,95·`h`; 0,4·`h`; радіуси 1× і 2,5× ребра; причина не задокументована | `src/application/services/advancing_front_mesher.py:608-617`, `src/application/services/advancing_front_mesher.py:632-656` |
| RCM multistart | [КОНСТАНТА] ліміт 20 кандидатів, крок 12, початковий пул максимум 15; причина не задокументована | `src/application/services/mesh_postprocessing.py:89-115` |
