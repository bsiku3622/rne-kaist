# RnE 서버 기동 방법

TensorBoard(학습 곡선)와 파일 서버(프로젝트 폴더 열람) 두 개를 띄우는 방법입니다.

- **프로젝트 루트**: `D:\School\KSA\RnE\rne-kaist\model`
- **접속 주소**: TensorBoard `http://10.8.0.2:6006`, 파일 서버 `http://10.8.0.2:8080`
- `10.8.0.2`는 이 PC의 VPN 인터페이스(`baeks-server-only-intranet`, 대역 `10.8.0.0/24`) 주소입니다.

---

## 1. 기동

PowerShell을 열고 아래를 통째로 붙여넣으면 됩니다.

```powershell
$root = "D:\School\KSA\RnE\rne-kaist\model"
Set-Location $root

$tb = Start-Process -FilePath "python" `
  -ArgumentList "-m","tensorboard.main","--logdir","runs","--host","0.0.0.0","--port","6006" `
  -RedirectStandardOutput "$root\tensorboard.log" -RedirectStandardError "$root\tensorboard.err" `
  -WindowStyle Hidden -PassThru

$fs = Start-Process -FilePath "python" `
  -ArgumentList "-u","-m","http.server","8080","--bind","10.8.0.2","--directory","$root" `
  -RedirectStandardOutput "$root\fileserver.log" -RedirectStandardError "$root\fileserver.err" `
  -WindowStyle Hidden -PassThru

"$($tb.Id)" | Out-File -Encoding utf8 "$root\.tensorboard.pid"
"$($fs.Id)" | Out-File -Encoding utf8 "$root\.fileserver.pid"

Write-Output "tensorboard PID = $($tb.Id)"
Write-Output "fileserver  PID = $($fs.Id)"
```

### 왜 `Start-Process`인가

터미널에서 `tensorboard --logdir runs &` 처럼 그냥 백그라운드로 돌리면 **셸 세션이 끝날 때 같이 죽습니다.** 실제로 아무 오류 로그도 남기지 않고 조용히 종료됩니다. `Start-Process`는 셸에서 분리된 독립 프로세스로 띄우므로 터미널을 닫아도 살아 있습니다.

`-RedirectStandardOutput`과 `-RedirectStandardError`는 **같은 파일을 지정할 수 없습니다.** 그래서 `.log`와 `.err`로 나눕니다. TensorBoard는 기동 메시지를 stderr로 내보내므로, 정상 기동 확인은 `tensorboard.err`를 봐야 합니다.

---

## 2. 확인

```powershell
netstat -ano | Select-String ":6006 |:8080 " | Select-String "LISTENING"
curl.exe -s -o NUL -w "tensorboard -> %{http_code}`n" http://10.8.0.2:6006/
curl.exe -s -o NUL -w "fileserver  -> %{http_code}`n" http://10.8.0.2:8080/
```

둘 다 `200`이면 정상입니다. TensorBoard가 인식한 run 목록은 이렇게 봅니다.

```powershell
curl.exe -s http://10.8.0.2:6006/data/runs
```

---

## 3. 종료

```powershell
$root = "D:\School\KSA\RnE\rne-kaist\model"
Stop-Process -Id (Get-Content "$root\.tensorboard.pid").Trim() -Force
Stop-Process -Id (Get-Content "$root\.fileserver.pid").Trim() -Force
```

PID 파일이 없거나 프로세스가 이미 죽은 경우엔 포트로 찾아서 죽입니다.

```powershell
Get-NetTCPConnection -LocalPort 6006,8080 -State Listen |
  Select-Object -ExpandProperty OwningProcess -Unique |
  ForEach-Object { Stop-Process -Id $_ -Force }
```

---

## 4. 주의사항

**`--host 0.0.0.0`을 빼면 안 됩니다.** TensorBoard의 기본값은 localhost 바인딩이라, 이 PC 밖에서는 접속할 수 없습니다.

**파일 서버는 반대로 `--bind 10.8.0.2`로 묶습니다.** `0.0.0.0`으로 열면 공인 IP와 집 LAN 쪽으로도 폴더 전체가 노출됩니다. VPN 인터페이스에만 묶어 두면 외부에서 직접 접근할 수 없습니다.

**빈 `runs\` 디렉터리를 지우지 마세요.** `--logdir` 대상이라 없으면 TensorBoard가 뜨지 않습니다. 학습 기록이 없어도 폴더 자체는 남겨 둬야 합니다.

**`archive\`는 ACL로 삭제가 차단돼 있습니다.** 연구 데이터 보존용입니다. 해제가 필요하면:

```powershell
icacls archive /remove:d "BAEKS-DESKTOP\bsiku"      # 해제
icacls archive /deny "BAEKS-DESKTOP\bsiku:(OI)(CI)(DE,DC)"   # 다시 잠금
```

---

## 5. 외부 도메인(`desktop.baeksikoo.com`) 연동

우분투 서버의 nginx가 이 PC로 프록시합니다. TensorBoard는 이미 `/`로 서빙 중이고, 파일 서버를 `/files`로 붙이려면 `desktop.baeksikoo.com` server 블록에 아래를 추가합니다. **아직 적용하지 않은 상태입니다.**

```nginx
location = /files {
    return 301 /files/;
}

location /files/ {
    allow 10.8.0.0/24;
    deny all;

    proxy_pass http://10.8.0.2:8080/;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Proto $scheme;

    # http.server가 디렉터리 요청에 절대 URL로 301을 쏘는데,
    # 경로에 /files 접두사가 없어서 그대로 두면 링크가 깨진다.
    proxy_redirect ~^https?://[^/]+/(.*)$ /files/$1;
}
```

적용은 `sudo nginx -t && sudo systemctl reload nginx`.

`proxy_pass` 끝의 슬래시가 핵심입니다. 이게 있어야 nginx가 `/files` 접두사를 떼고 백엔드에 넘깁니다. 파이썬 서버는 접두사를 모릅니다.

`allow`/`deny`는 nginx에 도달한 클라이언트의 `$remote_addr`를 봅니다. Cloudflare 같은 프록시가 앞단에 있으면 전부 403이 되므로, 그 경우 `set_real_ip_from`을 함께 설정해야 합니다. 차단이 실제로 되는지는 **VPN에 붙지 않은 외부 기기**에서 `https://desktop.baeksikoo.com/files/`가 403을 내는지로 확인하세요.

---

## 6. 알려진 제약

파이썬 `http.server`는 **단일 스레드**입니다. 31 MB짜리 `.npy`를 받는 동안 다른 파일 요청이 대기합니다. 불편해지면 `ThreadingHTTPServer`로 바꾸거나 nginx를 이 PC에 직접 설치하는 편이 낫습니다.

TensorBoard의 "실시간"은 push가 아니라 **폴링**입니다. 지연이 세 군데에 쌓입니다.

| 구간 | 기본값 | 조절 방법 |
|---|---|---|
| 학습 프로세스 → 디스크 | `flush_secs=120` (스칼라 10개 차면 즉시) | `SummaryWriter(..., flush_secs=10)` |
| 디스크 → TB 서버 | 5초 | `--reload_interval 2` |
| TB 서버 → 브라우저 | 30초 | UI 우측 상단 톱니바퀴 → Reload period |

현재 설정에서 체감 지연은 최대 40초 정도이고, 대부분 브라우저 폴링 탓입니다.
