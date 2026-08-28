# 아트 디렉션 파라미터 → 효과 맵

이 문서는 "어떤 요소가 무엇에 영향을 주는지"를 명시적으로 정리한 맵입니다.
미학적 튜닝을 할 때 각 노브가 어떤 부분의 실루엣/음영/거칠기에 기여하는지 확신 없이
만지는 blind 튜닝을 피하기 위해 작성했습니다. 각 요소는 `#6`~`#12` 이슈의 독립 함수/struct로
분리되어 있으므로, `?debug=regions` 오버레이로 해당 영역을 색으로 라벨링하며 기여를 격리해 볼 수 있습니다.

> 좌표계: `u` = 크레스트 선을 따라(0..1), `v` = 단면 프로파일을 따라(0=skirt .. 1=tongue 끝).
> 모든 "높이"는 curl 반경 단위(host 단위 아님). 실제 월드 크기는 `scale = waveCrestScale(u) * radius`로 곱해짐.

## 1. 단면 프로파일 — `waveProfile(v, curl, params)` (#6)

| 파라미터 / 구간 | 효과 | 시각 결과 |
|---|---|---|
| `WAVE_FACE_END` (0.40) | face 구간 끝 | face가 얼마나 긴지 |
| `WAVE_CREST_END` (0.52) | crest bulb 끝 | bulb 두께 |
| `WAVE_HOOK_END` (0.74) | hook 끝 | hook→tongue 전환점 |
| `lean = -0.30*sin(a*π)` (face) | face 오목도 | face가 안쪽으로 휘어 들어가는 정도 (concave) |
| `hookReach = (0.90+0.50*curlAmt)*(hg+0.5)` | hook 전방 돌출량 | lip이 앞으로 굽는 거리. **높이의 ~25–35% 목표** |
| `tipY` 하강 `(1.60+0.60*curlAmt)*(hg+0.4)` | tongue 하강량 | crest 최고점 아래로 떨어져 **barrel/음영 공간** 생성 |
| `curlAmt = 0.45+0.55*curl` | curl 민감도 | hook/tongue를 더 과장 |

## 2. 3D 크레스트 스켈레톤 — `CrestCurve` / `crestCentreline()` (#7, 미연결)

| 파라미터 | 효과 |
|---|---|
| `p0..p3` (제어점) | 크레스트 3D 곡선. 카메라 쪽으로 bow |
| `shape.y` (forwardBow) | 주능선이 카메라를 향해 휘는 양 |
| (예정) `bank`, `peakU` | 비대칭 플랭크 감쇠 |

> 아직 main.js가 `CrestCurve`를 패킹하지 않아 실루엣에 반영 안 됨. 연결 시 적용.

## 3. 폼 클로 ribbon — `FoamFinger` / `foamFingerPoint()` (#8, 미연결)

| 파라미터 | 효과 |
|---|---|
| `root` (rootU, rootV, parentIndex, generation) | 1차/2차 계층 위치 |
| `shapeA.z` (length) | finger 길이 |
| `shapeA.w` (base width) | ribbon 두께 (화면공간 최소폭과 곱해짐) |
| `shapeA.x` (tangent angle) | root에서의 방향 |
| `shapeB.x` (hook) | barrel 방향으로 굽는 정도 |

## 4. 목판 표면 — `classifyMaterial()` / `flowClass()` (#9, 미연결)

| 출력 | 효과 |
|---|---|
| `MAT_DEEP/BODY/FOAM/CLAW` | plate ID (FBM 아닌 기하 신호 기반) |
| `flowClass` (face/shoulder/hook/tongue) | 흐름선 방향 (T_art) |

## 5. Hero Wave 상태기계 — `heroArtParams(phase)` (#10, 부분연결)

`?hero=1&preset=kanagawa&seed=7&phase=0.64` 로 결정론적 프레임 고정.
phase 구간별 art curve:

| 곡선 | 위상 구간 | 의미 |
|---|---|---|
| `height` | 0→1 | 브레이커 전체 높이 |
| `faceConcavity` | 0.36→0.58 피크 | face 오목도 |
| `crestMass` | 0.36→0.58 피크 | crest 뭉탱이 |
| `hookReach` | 0.36→0.58 피크 | hook 전방 돌출 |
| `tongueDrop` | 0.36→0.58 피크 | tongue 하강 → barrel |
| `foamVisibility` | 0.58→0.72 피크 | 클로 가시성 |
| `ridgeBow` | 0.36→0.58 피크 | 주능선 bow |
| `secondaryRidge` | 0.36→0.58 피크 | 종속 능선 |

> 현재는 `?hero` 파싱만 되어 있고, breaker placement에 실제 연결은 안 됨(forcebreaker 경로로 결정론 캡처만 가능).

## 6. meso 파쇄 — `FractureFeature` / `fractureGate()` (#12, 미연결)

| 파라미터 | 효과 |
|---|---|
| `region` (centreU, centreV, width, depth) | 파쇄 위치/크기 (object-space 고정) |
| `shape.y` (skew) | 비대칭 방향 |
| `life` (birth/peak/death phase) | 위상에 따른 깊이 변화 |
| `fractureGate` | curvature×region×phase×envelope 곱 → 활성화 게이트 |

## 디버그 오버레이 (`?debug=regions`)
- face=파랑, crest bulb=흰색, hook=노랑, tongue=빨강 으로 단면 `v` 구간을 색칠.
- 어떤 기하 요소가 화면의 어디에 있는지 즉시 라벨링 → "어떤 요소가 무엇에 영향" 불안 해소.
