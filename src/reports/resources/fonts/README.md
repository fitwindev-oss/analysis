# Report Font Resources

PDF 리포트의 **결정적(deterministic) 한글 렌더링**을 위해 이 폴더에 폰트 파일을 배치합니다.

---

## 필요한 파일

아래 **두 명명 중 어느 쪽이든** 인식됩니다 (네이버 배포본이 둘 다 사용):

- `NanumGothic-Regular.ttf` **또는** `NanumGothic.ttf`
- `NanumGothic-Bold.ttf` **또는** `NanumGothicBold.ttf`

## 다운로드 방법

1. 네이버 나눔 글꼴 공식 배포 페이지 방문:
   https://hangeul.naver.com/font
2. **나눔 고딕 (Nanum Gothic)** 패키지 다운로드 (무료, 오픈 라이선스)
3. 압축 해제 후 **`NanumGothic-Regular.ttf`** 와 **`NanumGothic-Bold.ttf`** 두 파일만
   **이 폴더 (`src/reports/resources/fonts/`)** 에 복사

## 라이선스

나눔 글꼴은 **SIL Open Font License v1.1** 로 배포됩니다 — 상용 소프트웨어에 번들링
및 재배포가 자유롭게 허용됩니다. 단, 폰트 파일을 수정해 재배포할 경우 동일 라이선스
적용 의무가 있으니 원본 TTF 그대로 복사하세요.

SIL OFL 전문: https://scripts.sil.org/OFL

## 폴백 동작

위 파일이 없어도 앱은 정상 작동합니다:

1. **1순위** — 이 폴더의 번들 TTF
2. **2순위** — 시스템 설치 NanumGothic (`C:\Windows\Fonts\NanumGothic.ttf`)
3. **3순위** — Windows 기본 한글 폰트 **Malgun Gothic** (`malgun.ttf`)
4. **최종 폴백** — Helvetica (영문만 정상, 한글은 `□` 박스로 표시)

**최소 1/2/3 중 하나는 사용 가능해야** PDF에서 한글이 깨지지 않습니다. 대부분의
Windows PC에는 3번(Malgun Gothic)이 기본 설치되어 있어 문제없지만, 상용 배포 시엔
**이 폴더에 번들링**하는 것을 강력 권장합니다. 이유:

- 배포 대상 PC마다 다른 폰트 메트릭 → 줄바꿈/페이지 레이아웃 미묘하게 다름
- Windows 버전에 따라 Malgun Gothic 버전이 달라 시각적 일관성 저하
- 비-Windows 환경(Linux/macOS)에 배포 확장 시 대비

## 검증

폰트 배치 후 앱에서 PDF 한 장 내보내 보면:

- **정상**: 한글 제목/표가 깔끔히 렌더링
- **비정상 (Helvetica 폴백)**: 한글 모든 문자가 `□` 박스로 표시 → 2/3번 설치
  상태 확인 또는 이 폴더에 번들 재시도

## 관련 코드

- 폰트 경로 해석 로직: [`src/reports/fonts.py`](../../../fonts.py)
- 적용 시점: PDF 생성 시 1회 (`setup_korean_fonts()` 호출)
