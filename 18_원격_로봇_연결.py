"""
원격 두산 로봇 연결기 — 도메인/네임스페이스/IP/상태 자동 진단 + RViz 연결.

흐름:
  1) 캐시(~/.cache/doosan_remote.json) 있으면 "그대로 쓸까?" 묻고 즉시 진행
  2) 새 스캔: sniff(tcpdump) → 0 패킷이면 자동으로 brute(ros2 cli)로 fallback
  3) 각 도메인의 namespace 마다 진단:
       - 두산 노드 존재
       - joint_states Publisher count > 0 (가짜 vs 진짜 구분의 핵심)
       - robot_description 토픽 존재
       - host IP 파라미터
     → REAL / PUBLISHING / PARTIAL / EMPTY 4단계로 분류
  4) 표 형태로 나열하고 사용자가 idx 로 선택 (REAL 자동 우선)
  5) 토픽 진단: robot_description / joint_states / fixed_frame 결정
  6) RViz 설정 동적 생성 + rviz2 실행
  7) 모든 정보 캐시 저장

사용:
  python3 18_원격_로봇_연결.py
  python3 18_원격_로봇_연결.py --rescan
  python3 18_원격_로봇_연결.py --scan brute --brute-range 0-100
  python3 18_원격_로봇_연결.py --domain 10 --namespace dsr01
"""

import argparse
import atexit
import json
import os
import re
import select
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime


# ===== DDS / RTPS 상수 =====
RTPS_PORT_BASE = 7400
RTPS_DOMAIN_GAIN = 250
DDS_MCAST_GROUP = '239.255.0.1'

DOOSAN_TOPIC_HINTS = ('joint_states', 'dsr_robot', 'state',
                      'motion', 'gripper', 'flange_serial')
DOOSAN_NODE_HINTS = ('dsr_', 'robot_state', 'motion', 'gripper')

IP_PARAM_KEYS = ('host', 'ip', 'addr', 'address')
IP_RE = re.compile(r'\b(\d{1,3}(?:\.\d{1,3}){3})\b')

CACHE_PATH = os.path.expanduser('~/.cache/doosan_remote.json')


# ===== 공용 유틸 =====
def run(cmd, timeout=None, env=None):
    def _s(v):
        if v is None:
            return ''
        if isinstance(v, bytes):
            return v.decode(errors='replace')
        return v
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=env)
        return r.returncode, _s(r.stdout), _s(r.stderr)
    except subprocess.TimeoutExpired as e:
        return 124, _s(e.stdout), _s(e.stderr)
    except FileNotFoundError as e:
        return 127, '', str(e)


def env_with(domain_id):
    e = os.environ.copy()
    e['ROS_DOMAIN_ID'] = str(domain_id)
    e.pop('ROS_LOCALHOST_ONLY', None)
    return e


def stop_daemon():
    run(['ros2', 'daemon', 'stop'], timeout=5)


def detect_default_iface():
    _, out, _ = run(['ip', '-o', 'route', 'show', 'default'])
    m = re.search(r'dev\s+(\S+)', out)
    return m.group(1) if m else 'wlan0'


def ensure_sudo_for_sniff():
    """sudo 캐시 갱신 — sniff 모드 시작 전에 한 번 비번 받아두기.

    이미 캐시되어 있으면 비번 입력 없이 통과. 캐시 없으면 터미널에서 비번 묻기.
    반환: True (캐시 OK), False (실패 — brute 로 fallback)
    """
    print('  [sniff 준비] sudo 권한 캐시 확인 — 필요시 비번 입력')
    try:
        # capture_output 없이 호출 → 사용자 콘솔에 비번 프롬프트 띄움
        r = subprocess.run(['sudo', '-v'], timeout=120)
        if r.returncode == 0:
            return True
    except subprocess.TimeoutExpired:
        print('  [sniff 준비] 입력 대기 시간 초과')
    except Exception as e:
        print(f'  [sniff 준비] sudo 호출 실패: {e}')
    print('  [sniff 준비] sudo 실패 — brute 모드로 진행')
    return False


def is_plausible_ip(ip):
    if ip.startswith(('0.', '127.', '255.', '224.')):
        return False
    if ip in ('0.0.0.0', '255.255.255.255'):
        return False
    return True


# ===== 캐시 =====
def load_cache():
    if not os.path.exists(CACHE_PATH):
        return None
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_cache(data):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ===== 도메인 스캔: sniff =====
def sniff_domains(iface, timeout=8):
    if shutil.which('tcpdump') is None:
        print('  !! tcpdump 미설치 — sudo apt install tcpdump')
        return {}

    print(f'  [sniff] iface={iface}, timeout={timeout}s  (sudo 권한 필요)')
    print('         ※ 사전에 `sudo -v` 로 비밀번호 캐시해두면 끊김 없음')
    cmd = ['sudo', '-n', 'tcpdump', '-i', iface, '-nn', '-l',
           f'udp and dst host {DDS_MCAST_GROUP}']
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        print('  !! sudo 실행 실패')
        return {}

    by_domain = defaultdict(set)
    pat = re.compile(
        r'(\d+\.\d+\.\d+\.\d+)\.(\d+) > %s\.(\d+)' % re.escape(DDS_MCAST_GROUP)
    )
    started = time.time()
    packet_count = 0
    last_status = 0.0

    def _draw_status(final=False):
        elapsed = time.time() - started
        line = (f'\r  [sniff] {elapsed:5.1f}s / {timeout}s  '
                f'packets={packet_count:4d}  domains={len(by_domain)}')
        sys.stdout.write(line)
        if final:
            sys.stdout.write('\n')
        sys.stdout.flush()

    def _clear():
        sys.stdout.write('\r' + ' ' * 72 + '\r')
        sys.stdout.flush()

    try:
        while time.time() - started < timeout:
            remain = timeout - (time.time() - started)
            rlist, _, _ = select.select(
                [proc.stdout], [], [], min(0.3, max(0.05, remain))
            )
            now = time.time()
            if now - last_status >= 0.5:
                _draw_status()
                last_status = now
            if not rlist:
                if proc.poll() is not None:
                    break
                continue
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            packet_count += 1
            m = pat.search(line)
            if not m:
                continue
            src_ip = m.group(1)
            dst_port = int(m.group(3))
            if dst_port < RTPS_PORT_BASE:
                continue
            off = dst_port - RTPS_PORT_BASE
            if off % RTPS_DOMAIN_GAIN > 9:
                continue
            domain_id = off // RTPS_DOMAIN_GAIN
            if 0 <= domain_id <= 232:
                is_new = domain_id not in by_domain
                by_domain[domain_id].add(src_ip)
                if is_new:
                    _clear()
                    print(f'    ★ Domain {domain_id} 발견  ← PC {src_ip}')
    finally:
        _draw_status(final=True)
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    err = (proc.stderr.read() if proc.stderr else '') or ''
    if 'sudo:' in err.lower() or 'password' in err.lower():
        print('  !! sudo 비밀번호 필요 — `sudo -v` 후 재시도')

    return {d: ips for d, ips in by_domain.items()}


# ===== 도메인 스캔: brute-force =====
def bruteforce_domains(rng, timeout_each=2):
    print(f'  [brute] domain {rng.start}~{rng.stop - 1} 스캔 (각 {timeout_each}s)')
    found = {}
    for d in rng:
        stop_daemon()
        _, out, _ = run(['ros2', 'node', 'list'], timeout=timeout_each + 2,
                        env=env_with(d))
        nodes = [n.strip() for n in out.splitlines()
                 if n.strip() and not n.strip().startswith('/_ros2cli')]
        if nodes:
            found[d] = set()
            print(f'    ★ Domain {d}: {len(nodes)}개 노드 — '
                  f'{", ".join(nodes[:3])}{" ..." if len(nodes) > 3 else ""}')
    return found


# ===== 네임스페이스 자동 추출 =====
def list_namespaces(domain_id, timeout=3):
    stop_daemon()
    e = env_with(domain_id)
    _, nodes_out, _ = run(['ros2', 'node', 'list'], timeout=timeout + 2, env=e)
    _, topics_out, _ = run(['ros2', 'topic', 'list'], timeout=timeout + 2, env=e)

    ns_set = set()
    for line in (nodes_out + topics_out).splitlines():
        line = line.strip()
        if not line.startswith('/'):
            continue
        parts = line.lstrip('/').split('/')
        if len(parts) < 2:
            continue
        candidate = parts[0]
        rest = '/'.join(parts[1:])
        if (any(h in rest for h in DOOSAN_TOPIC_HINTS) or
                any(h in candidate for h in DOOSAN_NODE_HINTS) or
                any(h in rest for h in DOOSAN_NODE_HINTS)):
            ns_set.add(candidate)
    return sorted(ns_set)


# ===== 로봇 컨트롤러 IP 추출 =====
def detect_robot_ip(domain_id, namespace, timeout=4):
    e = env_with(domain_id)
    _, nodes_out, _ = run(['ros2', 'node', 'list'], timeout=timeout, env=e)
    nodes = [n.strip() for n in nodes_out.splitlines()
             if n.strip().startswith(f'/{namespace}/')]

    found_ips = set()
    for node in nodes:
        _, plist_out, _ = run(['ros2', 'param', 'list', node],
                              timeout=timeout, env=e)
        for line in plist_out.splitlines():
            p = line.strip()
            if not p:
                continue
            if any(k in p.lower() for k in IP_PARAM_KEYS):
                _, val_out, _ = run(['ros2', 'param', 'get', node, p],
                                    timeout=timeout, env=e)
                for m in IP_RE.finditer(val_out):
                    ip = m.group(1)
                    if is_plausible_ip(ip):
                        found_ips.add(ip)
    return sorted(found_ips)


# ===== 네임스페이스 종합 진단 =====
def diagnose_namespace(domain_id, namespace, timeout=4):
    """진짜 두산 로봇인지 빈 인터페이스인지 판정.

    REAL       : host_ips 검출 OR robot_description 토픽 존재
    PUBLISHING : joint_states Publisher count > 0 (host/desc 없어도)
    PARTIAL    : namespace 안에 노드는 있는데 joint_states publisher 없음
    EMPTY      : 토픽만 있고 publisher/노드 모두 없음 (가짜 인터페이스)
    """
    e = env_with(domain_id)
    info = {
        'js_pub_count': 0,
        'js_publishers': [],
        'has_robot_description': False,
        'host_ips': [],
        'ns_nodes': [],
        'status': 'EMPTY',
    }

    # 1) namespace 안의 노드
    _, nodes_out, _ = run(['ros2', 'node', 'list'], timeout=timeout, env=e)
    all_nodes = [n.strip() for n in nodes_out.splitlines() if n.strip()]
    info['ns_nodes'] = [n for n in all_nodes if n.startswith(f'/{namespace}/')]

    # 2) joint_states publisher 정보 (--verbose 로 Endpoint 블록 파싱)
    _, js_out, _ = run(
        ['ros2', 'topic', 'info', f'/{namespace}/joint_states', '--verbose'],
        timeout=timeout, env=e
    )
    m = re.search(r'Publisher count:\s*(\d+)', js_out)
    info['js_pub_count'] = int(m.group(1)) if m else 0
    if info['js_pub_count'] > 0:
        for blk in re.split(r'(?=Endpoint type:)', js_out):
            if blk.startswith('Endpoint type: PUBLISHER'):
                nm = re.search(r'Node name:\s*(\S+)', blk)
                ns_m = re.search(r'Node namespace:\s*(\S+)', blk)
                if nm:
                    label = nm.group(1)
                    if ns_m and ns_m.group(1) not in ('/', ''):
                        label = f'{ns_m.group(1).strip("/")}/{label}'
                    info['js_publishers'].append(label)

    # 3) robot_description 토픽 존재 여부
    _, rd_out, _ = run(
        ['ros2', 'topic', 'info', f'/{namespace}/robot_description'],
        timeout=timeout, env=e
    )
    info['has_robot_description'] = ('Unknown topic' not in rd_out
                                     and 'Type:' in rd_out)

    # 4) host IP from params
    info['host_ips'] = detect_robot_ip(domain_id, namespace, timeout=timeout)

    # 5) 종합 판정
    if info['host_ips'] or info['has_robot_description']:
        info['status'] = 'REAL'
    elif info['js_pub_count'] > 0:
        info['status'] = 'PUBLISHING'
    elif info['ns_nodes']:
        info['status'] = 'PARTIAL'
    else:
        info['status'] = 'EMPTY'
    return info


# ===== 모든 도메인 × 네임스페이스 → 로봇 후보 수집 =====
def gather_robots(found_domains):
    robots = []
    print()
    for d in sorted(found_domains.keys()):
        pc_ips = sorted(found_domains[d]) if found_domains[d] else []
        ns_list = list_namespaces(d)
        if not ns_list:
            print(f'  domain {d:>3}: 두산 ns 없음 — skip')
            continue
        for ns in ns_list:
            sys.stdout.write(f'  domain {d:>3}  ns={ns:<10}  진단 중...')
            sys.stdout.flush()
            diag = diagnose_namespace(d, ns)
            tag = {
                'REAL':       '✓ REAL',
                'PUBLISHING': '~ PUBLISHING',
                'PARTIAL':    '? PARTIAL',
                'EMPTY':      '✗ EMPTY',
            }[diag['status']]
            print(f'  → {tag}')
            if diag['host_ips']:
                last = ', '.join(ip.split('.')[-1] for ip in diag['host_ips'])
                print(f'              로봇 IP: {", ".join(diag["host_ips"])}  (끝번호 {last})')
            if diag['js_publishers']:
                print(f'              js publisher: {", ".join(diag["js_publishers"])}')
            if diag['has_robot_description']:
                print('              robot_description: ✓')

            robots.append({
                'domain': d,
                'namespace': ns,
                'robot_ips': diag['host_ips'],
                'pc_ips': pc_ips,
                'status': diag['status'],
                'js_pub_count': diag['js_pub_count'],
                'js_publishers': diag['js_publishers'],
                'has_robot_description': diag['has_robot_description'],
            })
    return robots


# ===== 토픽 진단 (RViz 설정용) =====
def list_topics(domain_id, timeout=5):
    stop_daemon()
    _, out, _ = run(['ros2', 'topic', 'list'], timeout=timeout, env=env_with(domain_id))
    return [t.strip() for t in out.splitlines() if t.strip()]


def detect_robot_description_topic(namespace, topics):
    candidates = [f'/{namespace}/robot_description', '/robot_description']
    topic_set = set(topics)
    for c in candidates:
        if c in topic_set:
            return c
    for t in topics:
        if t.endswith('/robot_description'):
            return t
    return None


def decide_fixed_frame(robot_desc_topic, namespace):
    if robot_desc_topic and robot_desc_topic.startswith(f'/{namespace}/'):
        return f'{namespace}/base_0'
    return 'base_0'


def detect_joint_states_topic(namespace, topics):
    candidates = [f'/{namespace}/joint_states', '/joint_states']
    topic_set = set(topics)
    for c in candidates:
        if c in topic_set:
            return c
    for t in topics:
        if t.endswith('/joint_states'):
            return t
    return f'/{namespace}/joint_states'


# ===== TF 실시간 검사 (rclpy 인라인) =====
_TF_INSPECT_CODE = r'''
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, DurabilityPolicy,
                       ReliabilityPolicy, HistoryPolicy)
from tf2_msgs.msg import TFMessage
import json
import time

rclpy.init()
node = Node("tf_inspector_inline")
se = set()
de = set()

def cb_s(msg):
    for t in msg.transforms:
        se.add((t.header.frame_id, t.child_frame_id))

def cb_d(msg):
    for t in msg.transforms:
        de.add((t.header.frame_id, t.child_frame_id))

qos = QoSProfile(depth=10,
                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                 reliability=ReliabilityPolicy.RELIABLE,
                 history=HistoryPolicy.KEEP_LAST)
node.create_subscription(TFMessage, "/tf_static", cb_s, qos)
node.create_subscription(TFMessage, "/tf", cb_d, 10)
end = time.time() + __TIMEOUT__
while time.time() < end:
    rclpy.spin_once(node, timeout_sec=0.1)
print("---TF_RESULT---")
print(json.dumps({
    "static": sorted([list(x) for x in se]),
    "dynamic": sorted([list(x) for x in de]),
}))
rclpy.shutdown()
'''


def inspect_tf_frames(domain_id, timeout=3):
    """별도 python3 -c 로 rclpy subscriber 띄워 /tf, /tf_static edge 수집."""
    code = _TF_INSPECT_CODE.replace('__TIMEOUT__', str(timeout))
    _, out, _ = run(['python3', '-c', code], timeout=timeout + 8,
                    env=env_with(domain_id))
    m = re.search(r'---TF_RESULT---\s*\n(.+)', out, re.DOTALL)
    if not m:
        return {'static': [], 'dynamic': []}
    try:
        return json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return {'static': [], 'dynamic': []}


def determine_fixed_frame_from_tf(tf_info, namespace, fallback):
    """TF edges 에서 root frame 추출 + namespace 우선 정렬.

    반환: (선택된 root, 후보 root 리스트)
    """
    all_edges = tf_info['static'] + tf_info['dynamic']
    if not all_edges:
        return fallback, []
    parents = {e[0] for e in all_edges}
    children = {e[1] for e in all_edges}
    roots = parents - children
    if not roots:
        roots = parents
    candidates = sorted(roots)

    # 우선순위: namespace 안의 base/world > 그냥 base_0/world > 그 외
    priority_groups = [
        [f'{namespace}/base_0', f'{namespace}/world'],
        ['base_0', 'world'],
    ]
    for group in priority_groups:
        for k in group:
            if k in candidates:
                return k, candidates
    # base 가 들어간 어떤 frame
    for c in candidates:
        if 'base' in c.lower():
            return c, candidates
    return candidates[0], candidates


# ===== RViz 설정 =====
RVIZ_TEMPLATE = """\
Panels:
  - Class: rviz_common/Displays
    Name: Displays
  - Class: rviz_common/Views
    Name: Views
Visualization Manager:
  Class: ""
  Displays:
    - Class: rviz_default_plugins/Grid
      Enabled: true
      Name: Grid
      Cell Size: 0.1
      Plane Cell Count: 20
    - Class: rviz_default_plugins/TF
      Enabled: true
      Name: TF
      Show Names: false
      Show Axes: true
    - Class: rviz_default_plugins/RobotModel
      Enabled: true
      Name: RobotModel
      Description Source: Topic
      Description Topic:
        Value: __DESC_TOPIC__
        Depth: 1
        Durability Policy: Transient Local
        History Policy: Keep Last
        Reliability Policy: Reliable
    - Class: rviz_default_plugins/PointCloud2
      Enabled: false
      Name: PointCloud2
      Topic:
        Value: /__NS__/camera/depth/color/points
        Depth: 5
        Reliability Policy: Best Effort
  Global Options:
    Fixed Frame: __FIXED_FRAME__
    Background Color: 48; 48; 48
  Tools:
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 2.0
      Focal Point:
        X: 0
        Y: 0
        Z: 0.5
      Name: Current View
      Pitch: 0.5
      Yaw: 0.785
"""


def make_rviz_config(namespace, robot_desc_topic, fixed_frame, out_path):
    cfg = RVIZ_TEMPLATE
    cfg = cfg.replace('__DESC_TOPIC__',
                      robot_desc_topic or f'/{namespace}/robot_description')
    cfg = cfg.replace('__FIXED_FRAME__', fixed_frame)
    cfg = cfg.replace('__NS__', namespace)
    with open(out_path, 'w') as f:
        f.write(cfg)
    return out_path


def launch_rviz(config_path, domain_id):
    if shutil.which('rviz2') is None:
        print('  !! rviz2 미설치')
        return None
    return subprocess.Popen(['rviz2', '-d', config_path],
                            env=env_with(domain_id),
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


# ===== [5/5] Motion (서보 풀고 움직이기) =====
def _confirm_motion(prompt, default=False, skip=False):
    if skip:
        return True
    sfx = '[y/N]' if not default else '[Y/n]'
    try:
        ans = input(f'  ⚠ {prompt} {sfx}: ').strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not ans:
        return default
    return ans in ('y', 'yes')


def detect_service_type(service_path, domain_id, timeout=5):
    """`ros2 service type` 으로 실제 srv 타입 조회. 없으면 None."""
    rc, out, _ = run(['ros2', 'service', 'type', service_path],
                     timeout=timeout, env=env_with(domain_id))
    if rc == 0 and out.strip():
        return out.strip().splitlines()[0].strip()
    return None


def call_service(service, srv_type, args_yaml, domain_id, namespace,
                 dry_run=False, timeout=40):
    """서비스 호출. srv_type 은 fallback — 실제 타입은 자동 조회로 보정.

    response 에 success=True 가 보이면 rc=124(timeout)이어도 성공으로 간주.
    """
    full = f'/{namespace}/{service}'
    # 실제 타입 자동 조회 (사용자 환경의 dsr_msgs2 버전 차이 흡수)
    actual = detect_service_type(full, domain_id) or srv_type
    if actual != srv_type and srv_type:
        print(f'  → {full}  (srv 타입 보정: {srv_type} → {actual})')
    else:
        print(f'  → {full}')
    print(f'    args: {args_yaml}')
    if dry_run:
        print('    [dry-run]')
        return True
    if not actual:
        print('    ✗ 서비스 타입 조회 실패 — 그 서비스 존재 안 함')
        return False
    cmd = ['ros2', 'service', 'call', full, actual, args_yaml]
    rc, out, err = run(cmd, timeout=timeout, env=env_with(domain_id))
    success_in_out = 'success=True' in out or 'success: true' in out.lower()
    if rc == 0 or (rc == 124 and success_in_out):
        snippet = (out.split('response:')[-1].strip()
                   if 'response:' in out else out.strip())
        tag = '✓' if rc == 0 else '⚠'
        suffix = '  (rc=124 timeout 났지만 응답에 success=True)' if rc == 124 else ''
        print(f'    {tag} {snippet[:200]}{suffix}')
        return True
    print(f'    ✗ rc={rc}  {(err or out).strip()[:300]}')
    return False


def takeover(domain_id, namespace, dry_run=False):
    """충돌해제 → protective stop 해제 → 서보 ON → autonomous."""
    print('\n=== 통제권 가져오기 ===')
    seq = [
        ('① 충돌(safe stop) 해제',
         'system/set_safe_stop_reset_type',
         'dsr_msgs2/srv/SetSafeStopResetType',
         '{reset_type: 1}'),
        ('② Protective stop 해제',
         'system/release_protective_stop',
         'dsr_msgs2/srv/ReleaseProtectiveStop',
         '{}'),
        ('③ 서보 ON (STANDBY)',
         'system/set_robot_state',
         'dsr_msgs2/srv/SetRobotState',
         '{robot_state: 0}'),
        ('④ Autonomous 모드',
         'system/set_robot_mode',
         'dsr_msgs2/srv/SetRobotMode',
         '{robot_mode: 1}'),
    ]
    ok = True
    for label, s, t, a in seq:
        print(f'\n {label}')
        ok = call_service(s, t, a, domain_id, namespace, dry_run=dry_run) and ok
    print()
    if ok:
        print('  ✓ 통제권 OK — motion 가능')
    else:
        print('  ⚠ 일부 실패 — 위 응답 확인')
    return ok


def movej_cmd(pos_list, domain_id, namespace, vel=30.0, acc=30.0, dry_run=False):
    if len(pos_list) != 6:
        print(f'  !! 6개 필요 (받음: {len(pos_list)})')
        return False
    args_yaml = (f'{{pos: {list(pos_list)}, vel: {vel}, acc: {acc}, '
                 'time: 0, radius: 0, mode: 0, blend_type: 0, sync_type: 0}}')
    print(f'\n=== movej {pos_list}  (vel={vel}, acc={acc}) ===')
    return call_service('motion/move_joint', 'dsr_msgs2/srv/MoveJoint',
                        args_yaml, domain_id, namespace,
                        dry_run=dry_run, timeout=30)


def interactive_motion_menu(domain_id, namespace, js_topic, dry_run=False):
    while True:
        print()
        print('=' * 64)
        flag = ' [DRY-RUN]' if dry_run else ''
        print(f'  Motion 메뉴 — domain={domain_id}, ns=/{namespace}{flag}')
        print('=' * 64)
        print('  [1] 현재 관절 상태 (read-only)')
        print('  [2] 통제권 가져오기 (충돌해제+서보ON+autonomous)')
        print('  [3] HOME 자세 (movej [0,0,90,0,90,0])')
        print('  [4] 사용자 입력 6개 관절각 movej')
        print('  [0] 종료')
        try:
            choice = input('선택: ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not choice or choice == '0':
            return
        if choice == '1':
            rc, out, err = run(['ros2', 'topic', 'echo', js_topic, '--once'],
                               timeout=8, env=env_with(domain_id))
            if rc == 0 and out.strip():
                print(out[:1200])
            else:
                print(f'  (못 받음) {err.strip()[:200]}')
        elif choice == '2':
            if dry_run or _confirm_motion('통제권 신호 보냅니다. 진행?'):
                takeover(domain_id, namespace, dry_run=dry_run)
        elif choice == '3':
            if dry_run or _confirm_motion('HOME 이동 진행?'):
                movej_cmd([0, 0, 90, 0, 90, 0], domain_id, namespace,
                          dry_run=dry_run)
        elif choice == '4':
            try:
                raw = input('  6개 관절각 (공백 구분, 예: 0 0 90 0 90 0): ').strip()
                pos = [float(x) for x in raw.split()]
            except (ValueError, EOFError, KeyboardInterrupt):
                print('  !! 입력 오류')
                continue
            if dry_run or _confirm_motion(f'movej {pos} 진행?'):
                movej_cmd(pos, domain_id, namespace, dry_run=dry_run)


def handle_motion(domain_id, namespace, js_topic, args):
    """[5/5] motion 단계 — argparse 플래그에 따라 자동 실행."""
    do_anything = any([args.auto, args.takeover, args.home, args.movej,
                       args.motion_menu])
    if not do_anything:
        return

    dry = args.motion_dry_run
    skip = args.yes

    print('\n' + '=' * 64)
    print(f'  [5/5] Motion — domain={domain_id}, ns=/{namespace}'
          f'{" [DRY-RUN]" if dry else ""}')
    print('=' * 64)

    if args.auto:
        if not (skip or _confirm_motion(
                '⚠ 실제 로봇에 통제권 → HOME 이동 자동 진행. OK?',
                default=False)):
            print('  취소')
            return
        if takeover(domain_id, namespace, dry_run=dry):
            movej_cmd([0, 0, 90, 0, 90, 0], domain_id, namespace,
                      vel=args.vel, acc=args.acc, dry_run=dry)
        return

    if args.takeover:
        if skip or _confirm_motion('통제권 가져오기 진행?'):
            takeover(domain_id, namespace, dry_run=dry)
    if args.home:
        if skip or _confirm_motion('HOME 이동 진행?'):
            movej_cmd([0, 0, 90, 0, 90, 0], domain_id, namespace,
                      vel=args.vel, acc=args.acc, dry_run=dry)
    if args.movej:
        try:
            pos = [float(x) for x in args.movej.split()]
        except ValueError:
            print('  !! --movej 파싱 실패')
            return
        if skip or _confirm_motion(f'movej {pos} 진행?'):
            movej_cmd(pos, domain_id, namespace,
                      vel=args.vel, acc=args.acc, dry_run=dry)
    if args.motion_menu:
        interactive_motion_menu(domain_id, namespace, js_topic, dry_run=dry)


# ===== bringup 자체 실행 (--launch-bringup) =====
def start_local_bringup(name='dsr01', host='110.120.1.18', mode='real',
                        model='e0509', wait=15, with_rviz=False):
    """자기 PC 에 두산 bringup 백그라운드 실행. Popen 반환 (실패 시 None).

    log → /tmp/dsr_bringup.log
    종료는 atexit 로 등록된 _cleanup_bringup() 가 처리.
    """
    launch_file = ('dsr_bringup2_rviz.launch.py' if with_rviz
                   else 'dsr_bringup2.launch.py')
    cmd = [
        'ros2', 'launch', 'dsr_bringup2', launch_file,
        f'name:={name}', f'host:={host}',
        f'mode:={mode}', f'model:={model}',
    ]
    print('\n' + '=' * 64)
    print('  [bringup] 자기 PC 에 두산 bringup 실행')
    print('=' * 64)
    print(f'  launch : {launch_file}')
    print(f'  name   : {name}')
    print(f'  host   : {host}  (두산 컨트롤러 IP)')
    print(f'  mode   : {mode}')
    print(f'  model  : {model}')
    print('  ※ E-stop 풀려있고 컨트롤러 전원 켜져있어야 함')

    log_path = '/tmp/dsr_bringup.log'
    try:
        log_f = open(log_path, 'w')
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        print('  !! ros2 launch 실행 실패 — ros2 CLI 또는 dsr_bringup2 미설치')
        return None

    print(f'  log    : {log_path}')
    print(f'  init 대기 {wait}초...')
    for i in range(wait):
        time.sleep(1)
        if proc.poll() is not None:
            print(f'\n  !! bringup 즉시 종료 (rc={proc.returncode}) — '
                  f'log 확인: {log_path}')
            return None
        sys.stdout.write(f'\r  init 대기 {i + 1:2d}/{wait}s')
        sys.stdout.flush()
    print(f'\n  [bringup] init 완료 (PID {proc.pid})')
    return proc


def _cleanup_bringup(proc):
    """atexit 콜백 — 스크립트 종료 시 bringup 도 같이 종료."""
    if proc is None or proc.poll() is not None:
        return
    print('\n[bringup] terminating...')
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print('  [bringup] SIGKILL')
        proc.kill()


# ===== 헬퍼 명령 출력 =====
def print_helpers(namespace, domain_id):
    bar = '=' * 64
    print('\n' + bar)
    print('  ★ 원격 제어 헬퍼 — 새 터미널에서 환경 먼저 적용')
    print(bar)
    print(f'  export ROS_DOMAIN_ID={domain_id}')
    print('  unset ROS_LOCALHOST_ONLY')
    print('  ros2 daemon stop')
    print(bar)
    print('\n# 1) 노드/토픽 확인')
    print('ros2 node list')
    print(f'ros2 topic list | grep ^/{namespace}/')
    print('\n# 2) 관절 상태 모니터')
    print(f'ros2 topic echo /{namespace}/joint_states')
    print('\n# 3) 충돌 해제')
    print(f'ros2 service call /{namespace}/system/set_safe_stop_reset_type \\')
    print('    dsr_msgs2/srv/SetSafeStopResetType "{reset_type: 1}"')
    print('\n# 4) 서보 ON')
    print(f'ros2 service call /{namespace}/system/set_robot_state \\')
    print('    dsr_msgs2/srv/SetRobotState "{robot_state: 0}"')
    print('\n# 5) Autonomous 모드')
    print(f'ros2 service call /{namespace}/system/set_robot_mode \\')
    print('    dsr_msgs2/srv/SetRobotMode "{robot_mode: 1}"')
    print('\n# 6) HOME movej')
    print(f'ros2 service call /{namespace}/motion/move_joint \\')
    print('    dsr_msgs2/srv/MoveJoint \\')
    print('    "{pos: [0,0,90,0,90,0], vel: 30, acc: 30, time: 0,'
          ' radius: 0, mode: 0, blend_type: 0, sync_type: 0}"')
    print('\n# 7) Protective stop 해제')
    print(f'ros2 service call /{namespace}/system/release_protective_stop \\')
    print('    dsr_msgs2/srv/ReleaseProtectiveStop "{}"')
    print(bar)


# ===== UI =====
def ask_yn(prompt, default=True):
    sfx = '[Y/n]' if default else '[y/N]'
    try:
        raw = input(f'{prompt} {sfx}: ').strip().lower()
    except EOFError:
        return default
    if not raw:
        return default
    return raw in ('y', 'yes')


def pick_robot(robots):
    if not robots:
        return None
    order = {'REAL': 0, 'PUBLISHING': 1, 'PARTIAL': 2, 'EMPTY': 3}
    sorted_robots = sorted(robots, key=lambda r: order.get(r['status'], 9))
    default = 0

    print('\n=== 진단 결과 — 발견된 로봇 ===')
    print('  idx │ status      │ domain │ ns        │ 로봇 IP (끝번호)               │ PC IP / publisher')
    print('  ────┼─────────────┼────────┼───────────┼────────────────────────────────┼──────────────────')
    for i, r in enumerate(sorted_robots):
        ips = r['robot_ips']
        if ips:
            last = ', '.join(ip.split('.')[-1] for ip in ips)
            ip_field = f'{", ".join(ips)}  ({last})'
        else:
            ip_field = '(미검출)'
        pcs = r['pc_ips']
        right = ', '.join(pcs) if pcs else '(none)'
        pub = ', '.join(r.get('js_publishers', []))
        if pub:
            right = f'{right}  pub={pub}'
        mark = ' ★' if i == default else '  '
        print(f'  [{i}]{mark}│ {r["status"]:<11} │  {r["domain"]:<5} │ {r["namespace"]:<9} │ '
              f'{ip_field:<30} │ {right}')
    print('\n  ✓ REAL = 실제 로봇  /  ~ PUBLISHING = 누군가 joint_states 만 보냄')
    print('  ? PARTIAL = 노드는 있는데 publisher 없음  /  ✗ EMPTY = 가짜 인터페이스 (조작불가)')

    if len(sorted_robots) == 1:
        print(f'\n  > 1개뿐 — 자동 선택: idx 0')
        return sorted_robots[0]
    try:
        raw = input(f'\n선택할 idx [{default}]: ').strip()
    except EOFError:
        return sorted_robots[default]
    if not raw:
        return sorted_robots[default]
    try:
        return sorted_robots[int(raw)]
    except (ValueError, IndexError):
        print('  !! 잘못된 입력')
        return None


# ===== main =====
def main():
    ap = argparse.ArgumentParser(
        description='원격 두산 로봇 — 도메인/IP/상태 자동 진단 + RViz 연결')
    ap.add_argument('--rescan', action='store_true',
                    help='캐시 무시하고 새로 스캔')
    ap.add_argument('--scan', choices=['sniff', 'brute', 'auto'], default='auto')
    ap.add_argument('--iface', default=None)
    ap.add_argument('--sniff-timeout', type=int, default=8)
    ap.add_argument('--brute-range', default='0-200')
    ap.add_argument('--domain', type=int, default=None)
    ap.add_argument('--namespace', default=None)
    ap.add_argument('--no-rviz', action='store_true')
    ap.add_argument('--rviz-config', default='/tmp/remote_doosan.rviz')
    # ----- [5/5] motion 옵션 -----
    ap.add_argument('--auto', action='store_true',
                    help='연결 후 통제권+HOME 자동 진행 (안전확인 1회)')
    ap.add_argument('--takeover', action='store_true',
                    help='연결 후 통제권만 (충돌해제+서보ON+autonomous)')
    ap.add_argument('--home', action='store_true',
                    help='연결 후 HOME 자세 movej')
    ap.add_argument('--movej', type=str, default=None,
                    help='연결 후 6개 관절각 movej (예: "0 0 90 0 90 0")')
    ap.add_argument('--motion-menu', action='store_true',
                    help='RViz 후 인터랙티브 motion 메뉴 진입')
    ap.add_argument('--vel', type=float, default=30.0)
    ap.add_argument('--acc', type=float, default=30.0)
    ap.add_argument('-y', '--yes', action='store_true',
                    help='모든 motion 확인 자동 yes (위험)')
    ap.add_argument('--motion-dry-run', action='store_true',
                    help='motion 명령 실제 호출 없이 출력만')
    # ----- bringup 자체 실행 (한 큐 처리) -----
    ap.add_argument('--launch-bringup', action='store_true',
                    help='자기 PC 에 두산 bringup 자체 실행 (한 명령으로 모두 처리)')
    ap.add_argument('--bringup-host', default='110.120.1.18',
                    help='두산 컨트롤러 IP (기본 110.120.1.18)')
    ap.add_argument('--bringup-model', default='e0509',
                    help='두산 모델 (기본 e0509)')
    ap.add_argument('--bringup-mode', default='real',
                    choices=['real', 'virtual'],
                    help='두산 모드 real/virtual (기본 real)')
    ap.add_argument('--bringup-name', default='dsr01',
                    help='두산 네임스페이스 (기본 dsr01)')
    ap.add_argument('--bringup-wait', type=int, default=15,
                    help='launch 후 init 대기 초 (기본 15)')
    ap.add_argument('--bringup-with-rviz', action='store_true',
                    help='dsr_bringup2_rviz.launch.py 사용 (자체 RViz 같이 띄움)')
    args = ap.parse_args()

    print('=== 원격 두산 로봇 연결기 ===\n')

    # ----- bringup 자체 실행 (--launch-bringup) -----
    if args.launch_bringup:
        bringup_proc = start_local_bringup(
            name=args.bringup_name,
            host=args.bringup_host,
            mode=args.bringup_mode,
            model=args.bringup_model,
            wait=args.bringup_wait,
            with_rviz=args.bringup_with_rviz,
        )
        if bringup_proc is None:
            return 1
        atexit.register(_cleanup_bringup, bringup_proc)
        if args.bringup_with_rviz:
            args.no_rviz = True
        if not args.rescan:
            print('  [bringup] --launch-bringup → --rescan 자동 적용')
            args.rescan = True

    domain_id = args.domain
    namespace = args.namespace
    robot_ips = []
    pc_ips = []
    status = None
    robot_desc_topic = None
    fixed_frame = None
    joint_states_topic = None

    # ----- 캐시 시도 -----
    cache = None if args.rescan else load_cache()
    if cache and domain_id is None and namespace is None:
        when = cache.get('discovered_at', '?')
        rips = cache.get('robot_ips', [])
        last = ', '.join(ip.split('.')[-1] for ip in rips) if rips else '?'
        cstatus = cache.get('status', '?')
        print(f'[캐시] domain={cache["domain_id"]}, '
              f'ns={cache["namespace"]}, '
              f'status={cstatus}, '
              f'robot 끝번호=[{last}]  ({when})')
        if ask_yn('  그대로 사용?', default=True):
            domain_id = cache['domain_id']
            namespace = cache['namespace']
            robot_ips = cache.get('robot_ips', [])
            pc_ips = cache.get('pc_ips', [])
            status = cache.get('status')
            robot_desc_topic = cache.get('robot_description_topic')
            fixed_frame = cache.get('fixed_frame')
            joint_states_topic = cache.get('joint_states_topic')
            print('  → 캐시 사용 (스캔 건너뜀)')

    # ----- 도메인 스캔 + 진단 -----
    if domain_id is None:
        print('\n[1/4] 도메인 스캔')
        found = {}
        if args.scan in ('sniff', 'auto'):
            iface = args.iface or detect_default_iface()
            sudo_ok = ensure_sudo_for_sniff()
            if sudo_ok:
                found = sniff_domains(iface, args.sniff_timeout)
            else:
                found = {}
            if not found and args.scan == 'auto':
                print('  → sniff 0 패킷, brute 모드 자동 전환')
                try:
                    lo, hi = (int(x) for x in args.brute_range.split('-'))
                except ValueError:
                    lo, hi = 0, 20
                found = bruteforce_domains(range(lo, hi + 1))
        else:
            try:
                lo, hi = (int(x) for x in args.brute_range.split('-'))
            except ValueError:
                lo, hi = 0, 20
            found = bruteforce_domains(range(lo, hi + 1))

        if not found:
            print('\n  !! 도메인 미발견.')
            return 1

        print('\n[2/4] 네임스페이스 진단 (두산 노드/IP/joint_states publisher 확인)')
        robots = gather_robots(found)
        if not robots:
            print('  !! 두산 후보 미발견')
            return 1

        # REAL 이 하나도 없으면 확장 스캔 제안 (0~232 까지)
        if not any(r['status'] == 'REAL' for r in robots):
            print('\n  ⚠ REAL 상태(실제 두산 bringup)인 항목이 없습니다.')
            print('    잡힌 건 모두 EMPTY/PARTIAL/PUBLISHING — 빈 인터페이스 가능성')
            try:
                lo, hi = (int(x) for x in args.brute_range.split('-'))
            except ValueError:
                lo, hi = 0, 200
            if hi < 200 and ask_yn(f'  domain {hi + 1}~200 까지 추가 스캔할까요?',
                                   default=True):
                extra = bruteforce_domains(range(hi + 1, 201))
                # 기존 결과와 머지
                for d, peers in extra.items():
                    found.setdefault(d, set()).update(peers)
                print('\n  [추가 진단] 새로 잡힌 도메인')
                extra_robots = gather_robots({d: found[d] for d in extra})
                robots.extend(extra_robots)

        chosen = pick_robot(robots)
        if chosen is None:
            return 1
        domain_id = chosen['domain']
        namespace = chosen['namespace']
        robot_ips = chosen['robot_ips']
        pc_ips = chosen['pc_ips']
        status = chosen['status']
        if status != 'REAL':
            print(f'\n  ⚠ 선택한 항목 status={status} — 조작 명령이 안 통할 수 있음')

    print(f'\n  → ROS_DOMAIN_ID={domain_id}, namespace=/{namespace}, status={status or "?"}')
    if robot_ips:
        print(f'  → 로봇 IP: {", ".join(robot_ips)}')
    os.environ['ROS_DOMAIN_ID'] = str(domain_id)
    os.environ.pop('ROS_LOCALHOST_ONLY', None)
    stop_daemon()

    # ----- 토픽 진단 + TF 실시간 검사 -----
    if robot_desc_topic is None or fixed_frame is None or joint_states_topic is None:
        print('\n[3/4] 토픽 진단 + TF 검사 (모델 정상 표시 자동 셋업)')
        topics = list_topics(domain_id)
        robot_desc_topic = detect_robot_description_topic(namespace, topics)
        joint_states_topic = detect_joint_states_topic(namespace, topics)
        print(f'  robot_description : {robot_desc_topic or "(미검출)"}')
        print(f'  joint_states      : {joint_states_topic}')

        # TF root 자동 검출 (rclpy 인라인 subscriber)
        sys.stdout.write('  TF 트리 검사 (3초)... ')
        sys.stdout.flush()
        tf_info = inspect_tf_frames(domain_id, timeout=3)
        n_s = len(tf_info['static'])
        n_d = len(tf_info['dynamic'])
        print(f'static={n_s} edges, dynamic={n_d} edges')

        fallback = decide_fixed_frame(robot_desc_topic, namespace)
        if n_s + n_d > 0:
            fixed_frame, root_candidates = determine_fixed_frame_from_tf(
                tf_info, namespace, fallback)
            print(f'  fixed_frame (TF root) : {fixed_frame}  ← 자동 결정')
            if len(root_candidates) > 1:
                others = [c for c in root_candidates if c != fixed_frame]
                print(f'  (다른 root 후보 : {", ".join(others)})')
        else:
            fixed_frame = fallback
            print(f'  fixed_frame (추정)   : {fixed_frame}')
            print('  ⚠ TF 가 안 흐름 — RViz 모델이 안 나오면 환경/bringup 확인')

    # ----- 캐시 저장 -----
    save_cache({
        'domain_id': domain_id,
        'namespace': namespace,
        'robot_ips': robot_ips,
        'pc_ips': pc_ips,
        'status': status,
        'robot_description_topic': robot_desc_topic,
        'fixed_frame': fixed_frame,
        'joint_states_topic': joint_states_topic,
        'discovered_at': datetime.now().isoformat(timespec='seconds'),
    })
    print(f'  [cache] saved → {CACHE_PATH}')

    # ----- RViz -----
    rviz_proc = None
    print('\n[4/4] RViz')
    if args.no_rviz:
        print('  --no-rviz 지정 — 시각화 생략')
    else:
        cfg = make_rviz_config(namespace, robot_desc_topic, fixed_frame,
                               args.rviz_config)
        print(f'  설정 → {cfg}')
        print(f'  토픽: {robot_desc_topic}, frame: {fixed_frame}')
        rviz_proc = launch_rviz(cfg, domain_id)
        if rviz_proc and not robot_desc_topic:
            print('  ⚠ robot_description 없음 → 모델 안 보일 것. status=REAL 아닌 환경.')

    print_helpers(namespace, domain_id)

    # ----- [5/5] motion — argparse 플래그 있으면 자동 실행 -----
    handle_motion(domain_id, namespace, joint_states_topic, args)

    print('\n[Enter] 로 종료...')
    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        if rviz_proc is not None and rviz_proc.poll() is None:
            rviz_proc.terminate()
            try:
                rviz_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                rviz_proc.kill()
    return 0


if __name__ == '__main__':
    sys.exit(main())
