# Hành trình huấn luyện tay robot nhấc chai bằng RL — Từ nền tảng tới lý thuyết sửa lỗi

> Tài liệu này kể lại **toàn bộ** hành trình: từ lúc dựng môi trường mô phỏng (Isaac Sim/Isaac Lab), thiết kế robot/scene/action-space, dạy REACH+GRASP, cho tới lúc phát hiện LIFT chưa bao giờ hoạt động (dù train hơn một tuần) và sửa tới khi đạt success_rate 57%. Phần đầu (Isaac Sim/Isaac Lab/kiến trúc env) là nền tảng đã ổn định trước khi việc "săn bug LIFT" bắt đầu, nên được mô tả như **kiến trúc**, không phải như một chuỗi bug — bug thật sự chỉ bắt đầu lộ ra từ Phần 2 trở đi. Chi tiết từng lệnh/log theo thời gian đã có trong `terminal_command.md` — tài liệu này là bản **tổng hợp lý thuyết**, để hiểu tầng sâu chứ không phải để tra lệnh.

---

# PHẦN 0 — Nền tảng: dựng môi trường mô phỏng và dạy REACH

## 0.1. Isaac Sim + Isaac Lab là gì, vì sao dùng chúng

**Isaac Sim** là bộ mô phỏng vật lý (dựa trên PhysX) của NVIDIA, cho phép giả lập robot, vật thể, va chạm, ma sát ở độ chính xác đủ để huấn luyện xong rồi triển khai lên robot thật (sim-to-real). **Isaac Lab** là framework RL xây trên nền Isaac Sim, cung cấp sẵn các "manager" chuẩn hoá một môi trường RL phức tạp thành các khối rõ ràng — đây chính là kiến trúc `ManagerBasedRLEnv` mà toàn bộ dự án dựa vào:

```
ManagerBasedRLEnv
 ├─ SceneCfg        — robot (USD), vật thể (chai, bát, bàn), ánh sáng, ground plane
 ├─ ActionsCfg      — cách policy điều khiển robot (khớp hay không gian thao tác)
 ├─ ObservationsCfg — vector trạng thái policy "nhìn thấy" mỗi bước
 ├─ RewardsCfg      — các reward term cộng lại thành phần thưởng mỗi bước
 ├─ TerminationsCfg — điều kiện kết thúc episode (thành công / thất bại / hết giờ)
 └─ EventsCfg       — sự kiện reset/randomize (vị trí chai, ma sát, ...)
```

Cài đặt (build từ source, không dùng bản standalone có sẵn — để kiểm soát version PhysX chính xác):
```bash
cd ~/isaacsim && ./build.sh                      # Isaac Sim, build từ source
cd ~/IsaacLab && uv venv --python 3.12 --seed env_isaaclab
source env_isaaclab/bin/activate
./isaaclab.sh -i 'newton,rl[rsl-rl],visualizer[newton]'
./isaaclab.sh -i 'rl[sb3]'                       # Stable-Baselines3 cho PPO
```

## 0.2. Robot, scene, và tư thế khởi đầu (start pose)

Robot OpenArm (7 khớp cánh tay + gripper 2 ngón) được nạp từ file USD (`v10.usd`) — đây chính là file đã bị phát hiện có bug vật lý ở Phần 2. Tư thế khởi đầu mỗi episode (`init_state.joint_pos`):

```python
joint1..3, 5..7 = 0.0      # cánh tay duỗi thẳng theo tư thế mặc định của URDF
joint4 = 1.57               # ~90°, gập khuỷu tay để đưa cổ tay hướng xuống bàn
finger_joint1/2 = 0.044     # gripper mở hoàn toàn (44mm)
```

Scene còn có: bàn (đã tắt va chạm vật lý — chỉ dùng làm mặt phẳng tham chiếu chiều cao), chai (RigidObject, vị trí random hoá mỗi episode qua `EventsCfg`), bát (RigidObject, dùng cho PLACE ở giai đoạn sau).

### Lý thuyết — vì sao tư thế khởi đầu quan trọng
Một tư thế khởi đầu **cố định, có hệ thống, gần với vùng làm việc mục tiêu** giúp policy học nhanh hơn nhiều so với khởi đầu ngẫu nhiên hoàn toàn — đây là một dạng **domain knowledge injection** đơn giản nhưng hiệu quả: ta không cần policy tự học "cách đứng dậy" mỗi episode, chỉ cần học phần việc thật sự (tiếp cận, kẹp, nhấc). Đánh đổi: policy học được có thể **quá khớp (overfit)** với đúng tư thế khởi đầu này — nếu triển khai lên robot thật ở tư thế khác, hành vi có thể không còn đúng.

## 0.3. Không gian hành động: OSC (Operational Space Control) thay vì điều khiển từng khớp

Thay vì để policy xuất ra 7 giá trị "mô-men mỗi khớp" (joint-space control), dự án dùng **OSC 6-DOF** (`AssistedOperationalSpaceControllerAction`, kiểu `pose_rel`): policy xuất ra một **độ dịch chuyển tương đối của đầu công tác (end-effector)** trong không gian 6 chiều (3 tịnh tiến + 3 xoay), bộ điều khiển OSC bên dưới tự tính ra mô-men khớp cần thiết để đạt được dịch chuyển đó (giải bằng động lực học ngược có tính tới quán tính, trọng lực).

### Lý thuyết — vì sao chọn không gian thao tác (task space) thay vì không gian khớp (joint space)
Nhiệm vụ ("đưa tay tới gần chai", "nhấc lên theo phương thẳng đứng") được định nghĩa tự nhiên trong **không gian Descartes** (vị trí/hướng của tay trong không gian 3D thật), không phải trong không gian góc khớp. Học trực tiếp trong không gian thao tác giúp:
- Reward shaping đơn giản hơn nhiều (khoảng cách Euclid tới mục tiêu, thay vì phải tính động học thuận từ 7 góc khớp).
- Không gian hành động nhỏ hơn về mặt "ý nghĩa" (6 chiều thao tác có ý nghĩa vật lý trực tiếp, dễ diễn giải/debug hơn 7 chiều góc khớp không tương quan tuyến tính với chuyển động tay).
- Bộ điều khiển OSC lo phần "vật lý khó" (bù trọng lực, quán tính, tránh kỳ dị động học) — policy chỉ cần học phần "chiến lược", không cần học lại cơ học robot từ đầu.

Đánh đổi: OSC cần một bộ điều khiển động lực học ngược đúng đắn bên dưới (đã có sẵn trong Isaac Lab), và có thể gặp khó ở các cấu hình gần kỳ dị (singularity) — không phải vấn đề gặp phải trong dự án này.

## 0.4. Thiết kế reward cho REACH — dạy "tới gần" bằng nhiều tầng tín hiệu chồng lên nhau

REACH (`_compute_reach_reward`) là giai đoạn ĐẦU TIÊN, và triết lý thiết kế của nó xứng đáng phân tích riêng vì nó là khuôn mẫu cho toàn bộ các giai đoạn sau (GRASP, PLACE):

```python
r_reach     = (1 - khoảng_cách) × 3 + exp(-8 × khoảng_cách) × 3      # "càng gần càng thưởng", 2 THANG ĐỘ (tuyến tính xa + mũ gần)
r_progress  = clamp((khoảng_cách_bước_trước − khoảng_cách_hiện_tại) × 40, -2, 5)   # thưởng ĐỘ CẢI THIỆN, không phải vị trí tuyệt đối
milestones  = thưởng rời rạc khi vượt qua các mốc khoảng cách (10cm, 8cm, 7cm, 6cm, 4cm, 3cm)
r_top_down  = thưởng hướng úp của kẹp càng thẳng xuống càng tốt, TĂNG MẠNH khi đã gần mục tiêu
r_hold      = thưởng giữ đủ lâu trong vùng "đã chạm" trước khi coi là sẵn sàng chuyển giai đoạn
```

### Lý thuyết — reward shaping nhiều tầng (dense reward tổng hợp từ sparse signal)
Nhiệm vụ gốc ("chạm được chai") vốn dĩ là **sparse reward** — chỉ có tín hiệu khi đạt đúng điều kiện, hầu như bằng 0 ở mọi nơi khác trong không gian trạng thái. Với reward thưa như vậy, một agent PPO khởi tạo ngẫu nhiên gần như **không bao giờ tình cờ chạm được mục tiêu** để có tín hiệu học đầu tiên (thảm hoạ "cold start" trong RL). Giải pháp — **reward shaping** — là thêm các tín hiệu **dày đặc** (dense) hướng dẫn từng bước nhỏ, miễn là các tín hiệu đó **không đổi thứ tự tối ưu của nhiệm vụ gốc** (định lý potential-based reward shaping của Ng, Harada, Russell 1999: shaping bằng đạo hàm của một hàm thế năng không đổi optimal policy).

Vài kỹ thuật cụ thể đáng chú ý làm khuôn mẫu:
- **Kết hợp tuyến tính + mũ** (`r_reach`) cho cùng một đại lượng: tuyến tính cho gradient ổn định khi còn xa, hàm mũ cho gradient dốc/nhạy khi đã gần (nơi độ chính xác quan trọng hơn).
- **Thưởng theo ĐỘ CẢI THIỆN** (`r_progress`) thay vì giá trị tuyệt đối: tránh được vấn đề "vị trí xuất phát ngẫu nhiên xa" bị thưởng thấp một cách không công bằng — chỉ quan tâm "đang tiến bộ hay không", ranh giới trên/dưới (`clamp`) ngăn chặn việc lợi dụng bước nhảy lớn bất thường (dạng exploit "teleport" nếu vật lý cho phép dịch chuyển đột ngột).
- **Milestone rời rạc**: tạo các "bậc thang" tín hiệu rõ ràng, giúp agent nhận biết đã tiến được bao xa so với các cột mốc có ý nghĩa, không chỉ có một gradient liên tục mơ hồ.
- **Cổng theo định hướng** (`orient_gate` nhân vào reward khoảng cách): chỉ thưởng "tới gần" đầy đủ khi hướng tiếp cận đã đúng — tránh agent học cách "lao vào" chai từ hướng sai chỉ để tối đa hoá phần thưởng khoảng cách.

> **Nguyên lý chung**: khi nhiệm vụ gốc có reward thưa, đừng thêm shaping một cách tuỳ tiện — luôn dùng shaping có tính "hướng dẫn tới đúng đích", đo đạc/kiểm tra bằng thực nghiệm rằng shaping không tạo ra một đường tắt (shortcut) có giá trị cao hơn nhiệm vụ thật (xem thêm phần "kinh tế học phần thưởng" bên dưới — chính REACH cũng từng bị khai thác theo kiểu "farm reward ở gần nhưng không hoàn thành", gọi là exploit "REACH-camping", xem Phần 1).

## 0.5. Vector quan sát (observation) — thiết kế cho khả năng mở rộng

Vector 26 chiều cố định, có cấu trúc rõ ràng theo khối: vị trí/vận tốc khớp (14), vị trí đầu công tác + vector hướng-tới-mục-tiêu (6, mục tiêu thay đổi theo giai đoạn), vector chai-tới-bát (3, chuẩn bị sẵn cho PLACE dù chưa dùng lúc chỉ có REACH/GRASP), trạng thái gripper (1), độ hở gầm bàn (1), và **một số thực hoá của giai đoạn hiện tại** (1, `0.0/0.5/1.0` cho REACH/GRASP/PLACE).

### Lý thuyết — vì sao giữ observation cố định kích thước qua nhiều giai đoạn phát triển
Trọng số mạng neural (đặc biệt lớp đầu vào) **gắn chặt với shape của observation**. Nếu thêm/bớt một chiều quan sát, checkpoint đã train **không thể nạp lại được** (`load_state_dict` báo lỗi shape mismatch) — mọi công sức train trước đó coi như hỏng, phải train lại từ đầu. Chính vì lý do này, quan sát "chai-tới-bát" đã được **thêm sẵn từ đầu**, dù mãi sau này mới có PLACE dùng tới — một dạng **thiết kế phòng thủ hướng-tới-tương-lai (forward-compatible design)**: chấp nhận vài chiều "chưa dùng tới" ngay bây giờ để tránh phải phá vỡ tương thích checkpoint sau này.

> **Nguyên lý chung**: trong RL nhiều giai đoạn (curriculum), nếu biết trước sẽ có giai đoạn tương lai cần thêm loại quan sát mới, **cân nhắc thêm chiều quan sát đó ngay từ đầu** (giá trị mặc định/không đổi khi chưa dùng tới) — rẻ hơn rất nhiều so với phải đánh đổi giữa "phá vỡ checkpoint cũ" và "không có đủ thông tin cho giai đoạn mới".

## 0.6. Thiết kế curriculum 3 giai đoạn: REACH → GRASP → PLACE trong CÙNG một episode

Thay vì train 3 policy riêng biệt cho 3 kỹ năng, dự án dùng **một state machine cấp cao** (`env._stage`) chuyển tuần tự REACH→GRASP→(PLACE) **trong cùng một episode**, với logic reward/assist khác nhau tuỳ giai đoạn nhưng cùng một policy, cùng một observation/action space.

### Lý thuyết — curriculum learning trong cùng một episode vs. train riêng từng giai đoạn
Cách này cho phép **transfer learning tự nhiên**: kỹ năng GRASP được học dựa trên chính điều kiện thật mà REACH để lại (không phải điều kiện lý tưởng do người thiết kế giả định) — ví dụ độ chính xác căn chỉnh thực tế sau REACH ảnh hưởng trực tiếp tới độ khó của GRASP. Đánh đổi lớn nhất — và là nguồn gốc của phần lớn bug ở Phần 2 — là các giai đoạn giờ **chia sẻ hạ tầng chung** (cùng reward-dispatch function, cùng biến assist scale, cùng action space), nên một thay đổi cho giai đoạn sau có thể vô tình ảnh hưởng giai đoạn trước nếu không cẩn thận cô lập (xem "lỗi kiến trúc — dùng chung núm vặn" ở Phần 2).

> **Nguyên lý chung**: curriculum trong-cùng-episode mạnh hơn train riêng lẻ, nhưng đòi hỏi kỷ luật cao hơn hẳn khi thêm giai đoạn mới — luôn tự hỏi "thay đổi này có đường nào chạy được ở giai đoạn KHÁC không, dù tôi không cố ý"? — và luôn regression-test giai đoạn cũ sau mỗi thay đổi cho giai đoạn mới.

---

# PHẦN 1 — Vì sao "nhấc" chưa từng có cơ hội xảy ra (Đợt điều tra 1)

Robot OpenArm cần thực hiện REACH → GRASP → LIFT (→ PLACE) một chai nước bằng RL (PPO). Sau train hàng triệu bước qua nhiều checkpoint, **success_rate = 0% tuyệt đối, `lift` đứng chết ở -0.012m, mọi episode đều timeout**. Việc tìm ra nguyên nhân đi qua **hai đợt điều tra lớn**, mỗi đợt gồm nhiều lớp lỗi ở các tầng khác nhau của hệ thống:

```
Tầng 4: Thuật toán học (PPO, credit assignment, curriculum)
Tầng 3: Reward shaping (khuyến khích đúng hành vi, kinh tế học γ-discount)
Tầng 2: Cơ chế điều khiển (assist/scripted action, state machine)
Tầng 1: Vật lý mô phỏng (actuator, joint, contact, mimic constraint)
```

Đây **không phải trùng hợp** mà là quy luật: một pipeline RL có nhiều tầng, lỗi có thể nằm ở bất kỳ tầng nào, trong khi triệu chứng ở tầng trên cùng luôn giống hệt nhau ("không học được gì", "reward không tăng"). Bug càng ở tầng thấp (vật lý) thì càng "câm lặng" — không traceback, không crash, chỉ đơn giản là **nhiệm vụ bất khả thi về cơ học**, nên không thuật toán nào cứu được dù train bao lâu.

4 agent đọc mã nguồn độc lập kết luận ngay từ đầu: **đây không phải vấn đề thiếu training**. Chín lỗi cụ thể (RC1-RC9) cộng năm lỗi tinh vi khác (N1-N5) xếp chồng lên nhau, đều đã được xác minh bằng trích dẫn mã nguồn.

## RC1 — Policy chưa từng được cấp quyền điều khiển hành động nhấc

### Triệu chứng
Assist (kịch bản hỗ trợ) được bật cho **toàn bộ** quá trình train Phase 2. Sau khi kẹp xong (latch), mọi hành động cánh tay do policy chọn đều bị **ghi giá trị 0** (`actions[mask] = 0.0`) trước khi thực thi.

### Lý thuyết — vi phạm giả định on-policy của PPO
PPO là thuật toán **on-policy**: nó lưu log-xác suất (log-prob) của **chính hành động policy đã chọn (sample)**, rồi dùng advantage để cập nhật đúng hành động đó. Nếu môi trường **âm thầm ghi đè hành động** (dù bằng 0 hay bằng một giá trị kịch bản khác), gradient cho kỹ năng đó **luôn bằng 0** — vì hành động policy chọn chưa từng ảnh hưởng tới kết quả. Đây là dạng lỗi tổng quát nghiêm trọng nhất trong cả investigation, xuất hiện lặp lại dưới nhiều hình thức (xem thêm ở Đợt 2, mục "residual thay vì override").

### Fix
Trả quyền điều khiển bằng **residual cộng thêm** thay vì ghi đè: `a_thực_thi = a_policy + w × bias`, với `w` là trọng số kịch bản (1 = hệt override để bootstrap, 0 = policy toàn quyền), giảm dần theo lịch train (anneal).

### Nguyên lý chung
> Bất kỳ khi nào bạn "giúp" một RL agent bằng scripted logic trong lúc train, hãy tự hỏi: **agent có đang được gradient hoá trên chính hành động nó chọn, hay trên một hành động nó chưa từng chọn?** Nếu là vế sau, dù reward "tốt" trên biểu đồ, không kỹ năng nào thực sự được học.

## RC2 — Thành công bị phạt (terminal value không được bootstrap đúng)

### Triệu chứng
Termination "success" không đánh dấu `time_out=True`, khiến Stable-Baselines3 coi đó là **kết thúc episode thật** (không phải do hết giờ) → `V(trạng thái cuối) = 0`, không có bonus bù đắp.

### Lý thuyết — bootstrapping giá trị terminal trong RL
Trong TD-learning/PPO, một episode kết thúc theo 2 kiểu khác nhau về mặt toán học:
- **Truncation** (`time_out=True`, hết giờ giả tạo): giá trị được **bootstrap** từ ước lượng của critic cho trạng thái tiếp theo (giả định "cuộc chơi vẫn tiếp diễn nếu không bị cắt ngang").
- **Termination thật** (nhiệm vụ thực sự kết thúc — thắng hoặc thua): `V=0` theo đúng định nghĩa (không còn tương lai để tính).

Nếu "thành công" là termination thật nhưng **không có phần thưởng tường minh cho chính sự kiện đó**, giá trị của nó trở thành: tổng shaping reward tích luỹ dọc đường + 0. So với "cứ đứng yên farm shaping reward tới hết giờ" (được bootstrap, có V dương), thành công trông **tệ hơn** về mặt số học thuần tuý — dù đó chính là mục tiêu ta muốn.

### Fix — cách SAI đã cân nhắc và bị loại
Thêm `time_out=True` cho success để nó được bootstrap giống truncation — **KHÔNG làm vậy**, vì sẽ bootstrap từ một trạng thái critic **chưa từng thấy** (trạng thái "vừa thành công"), phá vỡ phân loại truncation/termination, và biến giá trị thành công thành thứ không tune được trực tiếp.

### Fix đúng
Giữ success là termination thật, nhưng thêm **terminal bonus tường minh** (`terminal_success_bonus`, một reward term riêng đọc thẳng từ termination_manager) — biến giá trị của thành công từ "0 + shaping dọc đường" thành "bonus lớn, tường minh, tune được".

### Nguyên lý chung
> Khi một sự kiện "thắng/thua" kết thúc episode, **đừng dựa vào việc episode kết thúc để tạo động lực** — hãy thưởng/phạt tường minh cho chính sự kiện đó. Giá trị ngầm định từ V=0 gần như luôn sai lệch so với ý định của người thiết kế.

## RC3, RC4 — Cổng điều kiện bất khả thi & xung đột ngưỡng chồng chéo

`_grip_close_done()` đòi tiến độ đóng kẹp ≥96%, nhưng một cơ chế khác lại **chặn cứng** tiến độ đóng ở 75% (vì lo ngại đóng quá sâu làm lật chai) → điều kiện không bao giờ đúng, vĩnh viễn `False`. Hệ quả dây chuyền: một ngưỡng góc nghiêng cho phép đóng kẹp (4.5°) cao hơn ngưỡng góc nghiêng cho phép nhấc (4.0°) — vì cổng đóng-kẹp không bao giờ "xong" theo đúng định nghĩa, một biến trung gian bị kẹt ở giá trị sai, tạo ra vùng dao động nhấp nháy giữa hai ngưỡng.

### Nguyên lý chung
> Khi có **nhiều điều kiện liên quan tới cùng một đại lượng** (ở đây: "đã đóng kẹp đủ chưa") được định nghĩa ở nhiều nơi khác nhau (config mặc định vs. override runtime), chúng **phải nhất quán với nhau theo thiết kế**, không phải theo may rủi. Cách sửa bền vững: làm cho điều kiện tự nhận biết giới hạn thực tế (`done = min(ngưỡng_lý_tưởng, giới_hạn_thực_tế_đang_áp_dụng)`) thay vì đặt hai hằng số độc lập rồi hy vọng chúng khớp nhau.

## RC5 — Bug tự tham chiếu: biến bị ghi đè trước khi được đọc

Một cơ chế "gain thích ứng" cho lực nhấc đọc một biến trạng thái (`_prev_lift_bottle`) để tính "đã di chuyển bao nhiêu kể từ lần trước" — nhưng một hàm khác chạy **trước** nó trong cùng một bước lại ghi đè chính biến đó bằng giá trị **hiện tại**. Kết quả: "đã di chuyển" luôn tính ra 0, gain kẹt vĩnh viễn ở mức sàn.

### Nguyên lý chung
> Khi nhiều hàm trong cùng một bước thời gian **đọc và ghi cùng một biến trạng thái**, thứ tự gọi hàm quyết định đúng/sai — một lỗi kiểu này sẽ không bao giờ hiện trong test đơn lẻ từng hàm, chỉ hiện khi chạy đúng pipeline thật. Luôn vẽ rõ "hàm nào ghi, hàm nào đọc, ai chạy trước" khi có nhiều state được chia sẻ.

## RC6 — Mốc tham chiếu đo trước khi hệ thống "ổn định"

Độ cao nghỉ của chai (`_bottle_rest_z`) được lấy tại **thời điểm spawn**, trước khi vật lý kịp để chai rơi xuống bàn và ổn định — lệch 12mm so với mặt bàn thật. Mọi phép đo "đã nhấc bao nhiêu" đều dựa trên mốc sai này, khiến chai phải nhấc thêm 12mm "ảo" mới được tính là bắt đầu di chuyển — và ngưỡng thành công thực chất đòi hỏi nhấc cao hơn thật tới +42mm trên một chai cao 76.8mm.

### Nguyên lý chung
> Bất kỳ "giá trị nghỉ"/baseline nào được đo lúc khởi tạo (reset) **phải đợi hệ thống thực sự ổn định** (đủ số bước physics settle) trước khi lấy mẫu — đo quá sớm trong lúc vật vẫn đang rơi/dao động sẽ tạo ra một mốc tham chiếu méo, ảnh hưởng tới mọi phép đo tương đối phía sau nó.

## RC7 — `.any()` toàn cục khi scale lên nhiều môi trường song song

`if (một_điều_kiện).any(): đổi_ngưỡng_cho_tất_cả_env` — đúng khi test với 1-2 môi trường, nhưng khi train với 1024+ môi trường song song, **chỉ cần một môi trường** thoả điều kiện là toàn bộ 1023 môi trường khác cũng bị đổi ngưỡng theo — một dạng rò rỉ trạng thái giữa các môi trường lẽ ra phải độc lập tuyệt đối.

### Nguyên lý chung
> Trong RL huấn luyện song song (vectorized envs), **mọi phép toán trên tensor phải giữ đúng chiều per-env** (`mask[i]` cho từng env riêng), không được rút gọn bằng `.any()`/`.all()` trừ khi đó thực sự là ý định (ví dụ: dừng toàn bộ vì lỗi hệ thống). Bug này thường "ẩn náu" tốt vì code chạy đúng khi debug với 1-8 env, chỉỞ QUY MÔ LỚN mới lộ.

## Kinh tế học phần thưởng — vì sao train một tuần không tiến triển gì (γ-discounted return)

Đây là phát hiện **mang tính khai sáng nhất** của Đợt điều tra 1: chứng minh bằng số rằng "đứng yên farm reward" có giá trị kỳ vọng **cao hơn** "hoàn thành nhiệm vụ", nên PPO — vốn chỉ tối ưu đúng theo tín hiệu được cho — học chính xác điều **ngược lại** với ý định người thiết kế.

### Lý thuyết
PPO tối ưu **return chiết khấu** (discounted return): `G = Σ γᵗ · rₜ`, không phải reward tức thời. Với γ=0.99 và step 1/60s, chân trời "có trọng số đáng kể" chỉ khoảng 100 bước (≈1.67s) — sau đó đóng góp giảm dần theo cấp số nhân. Điều này có nghĩa: **một dòng reward nhỏ nhưng liên tục** trong 100 bước có thể có giá trị chiết khấu lớn hơn hẳn **một phần thưởng một-lần** dù phần thưởng đó "nhìn có vẻ lớn hơn" theo giá trị tuyệt đối.

### Số liệu đo được thực tế (baseline, trước khi sửa)
| Trạng thái | Reward thô/bước | Giá trị chiết khấu (γ=0.99, 100 bước) |
|---|---|---|
| **Đứng yên ôm chai** (farm shaping reward liên tục) | +97 | **≈ +162** |
| **Nhấc thành công** (giữ 5 bước rồi episode kết thúc, `V=0`, không bonus) | +341 (chỉ trong 5 bước) | **≈ +27.8** |

**Đứng yên farm reward có giá trị gấp 5.8 lần hoàn thành nhiệm vụ.** Đường cong `ep_rew_mean` tăng đều trong suốt tuần train chính là bằng chứng: agent đang học cách "camp" lâu hơn — nghịch biến hoàn toàn với thành công thật.

### Sau khi sửa (thêm terminal bonus, camp decay, tipped penalty — γ=0.99)
| Trạng thái | Giá trị |
|---|---|
| Camp (>60 bước sau latch, shaping đã suy giảm về 0, phạt thời gian nhẹ) | **−11.7** |
| Lật chai (terminal penalty) | **−30.0** |
| Nhấc thành công (rise 25 bước + hold 5 bước + bonus lớn chiết khấu) | **+72.8** |

Lợi thế của "nhấc" so với "camp" đảo chiều từ **−134 sang +84.5** — đổi dấu hoàn toàn, cùng độ lớn ở hướng đúng.

### Nguyên lý chung
> **Không bao giờ thiết kế reward bằng trực giác "cái này nghe có vẻ nên thưởng nhiều hơn".** Luôn tính tay (hoặc bằng script) **giá trị chiết khấu kỳ vọng** của từng "chiến lược" khả dĩ (kể cả những chiến lược lười biếng/gian lận bạn không cố ý cho phép) trước khi tin vào một hàm reward. Nếu chiến lược gian lận có giá trị cao hơn chiến lược đúng, agent — nếu đủ mạnh — **sẽ luôn tìm ra nó**, không phải vì nó "thông minh xấu tính" mà vì nó đang làm đúng việc của một bộ tối ưu hoá.

## Thiết kế lại: state machine thay cho một mớ cổng điều kiện rời rạc

Thay vì hàng chục cờ/ngưỡng độc lập kiểm tra lại **mỗi bước**, toàn bộ logic nhấc được viết lại thành một **state machine 3 trạng thái**: `IDLE → RISING → HOLDING`.

### Lý thuyết — vì sao "kiểm tra lại điều kiện mỗi bước" gây nhấp nháy
Nếu điều kiện bắt đầu một hành động được đánh giá lại **liên tục trong khi hành động đang diễn ra**, một dao động nhỏ ở ranh giới điều kiện (do nhiễu vật lý, dao động quán tính) sẽ làm hành động **bật/tắt liên tục** — lệnh "nhấc" chỉ tồn tại được vài chục mili-giây trước khi bị huỷ bởi chính điều kiện đã cho phép nó bắt đầu. Tính chất quyết định của state machine đúng: **một khi đã chuyển sang trạng thái "đang làm" (RISING), KHÔNG đánh giá lại điều kiện bắt đầu** — chỉ có điều kiện huỷ (abort) thực sự mới được phép dừng nó giữa chừng, và điều kiện huỷ phải **khác biệt và khắt khe hơn** điều kiện bắt đầu (tránh đúng vùng biên gây dao động).

### Nguyên lý chung
> Khi một hành vi cần "cam kết" trong nhiều bước (nhấc, mang đi, hạ xuống...), đừng lập trình nó như "if điều_kiện: làm_một_bước" — hãy dùng state machine tường minh với: (1) điều kiện BẮT ĐẦU chỉ đánh giá lúc IDLE, (2) điều kiện HUỶ khác và khắt khe hơn, chỉ áp dụng khi đang "đang làm", (3) không bao giờ đánh giá lại điều kiện bắt đầu trong lúc đang thực thi.

---

# PHẦN 2 — Vì sao nhấc "thỉnh thoảng khả thi" vẫn không bao giờ thành công thật (0% → 57%, Đợt điều tra 2)

Sau khi Đợt 1 khiến việc nhấc **khả thi về mặt cơ chế điều khiển**, robot vẫn **0% thành công** khi train thật. Người dùng trực tiếp quan sát: *"hai ngón tay trái/phải không đối xứng khi kẹp"* — một quan sát trực giác đơn giản dẫn tới phát hiện lớn nhất toàn bộ dự án.

## Lỗi vật lý #1 — Mimic joint quá "mềm" (lò xo ảo yếu)

Trong PhysX, một cặp khớp "phải giống nhau" (ví dụ hai ngón đối xứng) có thể nối bằng **PhysxMimicJointAPI** — hiện thực bằng một **lò xo-giảm chấn ảo**, không phải ràng buộc cứng. Hai tham số quyết định độ "cứng": `naturalFrequency` (phản ứng nhanh/chậm) và `dampingRatio` (0 = dao động không tắt, 1 = tới hạn/nhanh nhất không dội, >1 = êm nhưng chậm). Giá trị gốc `naturalFrequency=25, dampingRatio=0.005` — cực mềm, gần như không giảm chấn. Khi khớp chủ di chuyển nhanh, khớp phụ không theo kịp — độ trễ **tỉ lệ với tốc độ đóng**, đo được lệch tới 52mm khi đóng theo ramp thật (đường kính chai chỉ 43mm).

**Fix**: vá trực tiếp vào file USD (`naturalFrequency=200, dampingRatio=1.0`) — sửa ở đúng tầng asset vật lý, không phải runtime.

> **Nguyên lý**: khi hai vật "phải giống nhau" trong mô phỏng, luôn kiểm tra đó là ràng buộc cứng hay lò xo ảo. Lò xo ảo luôn trễ tỉ lệ tốc độ — đừng đánh giá bằng mắt ở tốc độ chậm rồi kết luận "ổn".

## Lỗi vật lý #2 — Thiếu quyền lực actuator (ai thực sự "cầm lái" khớp?)

Sau khi sửa lỗi #1, hai ngón vẫn lệch **dưới tải thật** (khi ép vào chai) dù đóng chậm trong không khí thì ổn. Đo bằng cách đẩy lực ngoài: một ngón phục hồi vị trí yếu hơn ngón kia **43%**.

### Lý thuyết
Một khớp chỉ nhận lệnh PD (Proportional-Derivative control) nếu nó được liệt kê trong một `ImplicitActuatorCfg` — đây chính là "ai được cấp quyền lực điều khiển". Khớp không nằm trong actuator group nào chỉ còn dựa vào ràng buộc mimic để "được kéo theo" — dưới tải bên ngoài, mimic (dù đã cứng) vẫn **yếu hơn hẳn** một bộ điều khiển PD chủ động.

### Cạm bẫy khi sửa
Thử đơn giản — thêm khớp vào actuator group — làm **tệ hơn**: khớp mới nhận PD gain thật nhưng **target của nó chưa từng được cập nhật** (code chỉ ghi lệnh cho khớp thứ nhất). PD với target "đứng yên" tạo lực **cản trở** chính chuyển động mimic đang cố kéo nó theo.

**Fix đúng (2 mảnh ghép bắt buộc đi cùng nhau)**: (1) cấp quyền lực PD thật cho cả hai khớp, VÀ (2) ghi target tường minh, giống hệt nhau, cho cả hai khớp mỗi bước. Xác nhận cách làm bằng cách đối chiếu với repo tham khảo chính thức của nhà sản xuất robot.

> **Nguyên lý**: "cấp quyền lực" và "ra lệnh" là hai việc tách biệt, luôn phải đi cùng nhau. Cấp quyền lực mà không ra lệnh đúng tạo ra một bộ điều khiển chống lại chính mục tiêu của bạn.

## Lỗi reward — Suy giảm phần thưởng neo sai mốc thời gian (v5 của cùng một họ bug với RC2)

Train dài hạn: `latch_rate` giảm dần đều về 0 trong khi `ep_rew_mean` **tăng** — chữ ký kinh điển của "agent đang tối ưu đúng theo tín hiệu sai" (cùng bản chất với phần "kinh tế học phần thưởng" ở Đợt 1, nhưng lần này ẩn kỹ hơn — nó chỉ lộ ra SAU KHI vật lý đã được sửa, vì trước đó latch/lift gần như không bao giờ xảy ra nên cơ chế decay chưa từng có cơ hội gây hại).

### Nguyên nhân
Một cơ chế "suy giảm phần thưởng nếu đứng yên quá lâu" (chống farm reward — ý tưởng đúng, xem RC2/kinh tế học ở Đợt 1) bị neo nhầm mốc: đếm từ *lúc vào giai đoạn GRASP* thay vì từ *lúc thực sự đã kẹp xong*. Thời gian tiếp cận + đóng kẹp đo thật là 136-371 bước, trong khi decay được thiết kế về 0 sau 200 bước — **hết hạn gần như ngay trước khi kịp kẹp xong**, giết chết toàn bộ phần thưởng định hướng đúng lúc cần nhất (trong lúc thử nhấc).

**Fix**: bộ đếm mới chỉ tăng khi điều kiện thật đã đạt (đã latch), reset khi mất latch.

> **Nguyên lý**: khi thiết kế decay chống lười biếng, luôn hỏi "decay này đếm từ đâu, và mốc đó có đúng là lúc hành vi XẤU bắt đầu có thể xảy ra không?" — nếu mốc đếm sớm hơn cả công việc HỢP LỆ, bạn đang phạt nhầm. **Luôn đo thời gian thật cần thiết bằng thực nghiệm**, đừng suy luận.

## Lỗi kiến trúc — Dùng chung một "núm vặn" cho hai kỹ năng không liên quan

Sau khi dạy kỹ năng MỚI (mang chai qua bát), kỹ năng CŨ đã ổn định (tiếp cận + kẹp) bất ngờ thoái hoá dù không ai đụng code của nó — `grasp_rate` sập từ 0.99 xuống 0.40.

### Nguyên nhân
Một biến điều khiển duy nhất (`assist scale`) được tái sử dụng cho NHIỀU mục đích khác nhau (hỗ trợ tiếp cận, hỗ trợ kẹp, hỗ trợ nhấc, hỗ trợ mang) cộng với một điểm **return sớm** gắn liền với chính biến đó. Khi lịch curriculum đưa biến về 0 để ép policy tự học kỹ năng MỚI, nó tắt luôn hỗ trợ cho kỹ năng CŨ — dù về logic hai kỹ năng độc lập hoàn toàn.

**Fix**: tách hai biến — một biến **cố định** bảo vệ kỹ năng cũ, một biến **anneal dần về 0** chỉ áp dụng cho kỹ năng đang được dạy. Biến mới phải tự động "im lặng" (bằng biến cũ) khi không ai chủ động dùng cơ chế mới, để không phá vỡ bất kỳ cách gọi cũ nào.

### Kết quả đo được
| | Trước fix | Sau fix |
|---|---|---|
| `grasp_rate` (rất lâu sau khi assist về 0) | Sập 0.99 → 0.40 | Giữ vững 0.99-1.0 |
| `latch_rate` | Sập về 0.00 | Ổn định 0.40-0.58 |

> **Nguyên lý**: trước khi cho hai cơ chế dùng chung một biến/một điểm thoát sớm, hãy vẽ rõ biến đó ảnh hưởng tới NHỮNG GÌ — nếu chúng không thực sự nên cùng thay đổi theo cùng lịch trình, hãy tách. Chi phí tách sớm luôn rẻ hơn chi phí gỡ một collapse không rõ nguyên nhân.

---

## Bài học phương pháp luận chung (áp dụng cho mọi dự án RL/robotics khác)

1. **Đo trước khi đoán, luôn luôn.** Không hằng số quan trọng nào (decay steps, ngưỡng lực, thời gian cần thiết, giá trị chiết khấu của một chiến lược) nên được đặt bằng "cảm giác hợp lý" — mọi con số then chốt trong dự án này đều bắt nguồn từ một script đo trực tiếp, không phải suy luận lý thuyết.
2. **Tính tay giá trị chiết khấu (γ-discounted return) của MỌI chiến lược khả dĩ**, kể cả các chiến lược "gian lận" bạn không cố ý cho phép — trước khi tin vào một hàm reward. Nếu gian lận có giá trị cao hơn, agent đủ mạnh sẽ luôn tìm ra nó.
3. **Thay đổi một biến một lúc, verify riêng từng cái.** Gộp nhiều thay đổi (vật lý + reward, hoặc kỹ năng cũ + kỹ năng mới) vào cùng một lần train khiến kết quả xấu **không thể quy được nguyên nhân**.
4. **Regression gate sau MỌI thay đổi có rủi ro**, kể cả khi nhắm tới một tính năng khác — một thay đổi cho kỹ năng MỚI hoàn toàn có thể phá kỹ năng CŨ nếu chúng chia sẻ hạ tầng (state machine, actuator, assist scale...).
5. **Agent không "lười biếng" hay "gian lận" theo nghĩa xấu tính — nó đang tối ưu chính xác theo tín hiệu bạn đưa cho nó.** Khi hành vi học được "sai", đừng đổ lỗi cho thuật toán — hãy tìm lại đúng tầng (vật lý/cơ chế/reward/thuật toán) nơi tín hiệu bị lệch.
6. **Khi bế tắc, đối chiếu với một implementation tham khảo đã kiểm chứng** — nhiều khi câu trả lời không phải "nghĩ ra ý tưởng mới" mà là "làm đúng như người khác đã làm đúng".
7. **Tin vào trực giác vật lý của chính mình khi nó mâu thuẫn với số liệu train.** Đột phá lớn nhất của toàn bộ Đợt điều tra 2 bắt đầu từ một quan sát bằng mắt rất đơn giản — "hai ngón tay không đối xứng" — không phải từ một con số trong log, và nó xuất hiện đúng lúc người viết code đã đề xuất "chấp nhận đây là giới hạn vật lý, dừng lại" — bị bác bỏ, và đúng là bác bỏ đúng.

---

## Phụ lục — Ví dụ tái diễn của lỗi #4, xảy ra ngay sau khi tài liệu này được viết

Ngay sau khi hoàn thành phần lý thuyết ở trên, đúng lớp lỗi #4 ("dùng chung một núm vặn") tái diễn dưới một hình thức khác — một minh chứng sống rằng nguyên lý chung áp dụng được ngoài phạm vi ví dụ gốc.

**Việc cần làm**: đo lại hình học bát thật (bằng `UsdGeom.BBoxCache`) vì phát hiện tâm bát bị lệch ~8cm so với điểm neo (root) đang dùng — nghe có vẻ "chỉ ảnh hưởng logic mang-chai-tới-bát (PLACE)". Sửa công thức tính vị trí tương đối "chai-tới-bát" để dùng tâm bát đã sửa đúng thay vì điểm neo lệch.

**Regression xảy ra ngay lập tức**: `grasp_rate` của một checkpoint đã train từ lâu sập từ 0.83 xuống 0.47.

**Nguyên nhân**: đại lượng "chai-tới-bát" đó **không chỉ dùng nội bộ cho PLACE** — nó còn là một phần của **vector quan sát (observation)** đưa vào chính sách ở MỌI giai đoạn (không chỉ PLACE). Đổi công thức tính nó — dù giữ nguyên **kích thước** (shape) của vector quan sát — vẫn tương đương đổi **nội dung** quan sát: chính sách đã train quen với một phân bố giá trị nhất định ở đúng vị trí đó, đổi công thức làm phân bố dịch đi ~8cm, đủ để phá vỡ hành vi đã ổn định.

**Bài học rút ra thêm** (mở rộng nguyên lý #4 ở trên): "núm vặn dùng chung" không chỉ là một biến điều khiển (assist scale) — nó có thể là **bất kỳ đại lượng trung gian nào được tính một lần rồi dùng ở nhiều chỗ khác nhau**, bao gồm cả một thành phần của vector quan sát. Trước khi sửa MỘT hàm tính toán trạng thái dùng chung (ở đây là `compute_state()`), luôn tự hỏi: **giá trị này có lọt vào observation/action mà một mạng neural ĐÃ TRAIN đang phụ thuộc vào không?** — nếu có, "chỉ ảnh hưởng tính năng mới" là một giả định cần kiểm chứng bằng regression gate, không phải điều hiển nhiên. Cách sửa an toàn: tách hai đại lượng — một giữ NGUYÊN công thức cũ để nuôi observation, một biến MỚI dùng công thức đúng chỉ cho logic nội bộ của tính năng mới.
