"""
원격 두산 로봇 조작기 — 18 스크립트가 만든 캐시로 곧장 명령 전송.

흐름:
  1) ~/.cache/doosan_remote.json 캐시 읽음 (18_원격_로봇_연결.py 결과)
  2) 메뉴 또는 argparse 옵션으로 동작 선택
  3) 안전 확인 후 ros2 service call 실행

기본 시퀀스 (강의 실연):
  ① 1번 — 현재 관절 상태 확인 (read-only, 안전)
  ② 2번 — 통제권 가져오기 (충돌해제 + 서보ON + autonomous)
  ③ 3번 — HOME 자세로 이동
  ④ 4번 — 사용자 입력 6개 관절각으로 이동

사용:
  python3 19_원격_조작.py              # 인터랙티브 메뉴 (권장)
  python3 19_원격_조작.py --status     # 상태만 출력
  python3 19_원격_조작.py --takeover   # 통제권만 가져오기
  python3 19_원격_조작.py --home
  python3 19_원격_조작.py --movej "0 0 90 0 90 0"
  python3 19_원격_조작.py --dry-run --home   # 명령만 출력 (실제 호출 X)
"""
import argparse
import importlib.util
import os
import sys


_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 18 스크립트의 run, env_with, load_cache 등 재사용
p18 = _load('p18', '18_원격_로봇_연결.py')


# ===== 서비스 호출 =====
def call_service(service, srv_type, args_yaml, domain_id, namespace,
                 dry_run=False, timeout=20):
    full = f'/{namespace}/{service}'
    cmd = ['ros2', 'service', 'call', full, srv_type, args_yaml]
    print(f'  → {full}')
    print(f'    args: {args_yaml}')
    if dry_run:
        print('    [dry-run — 실제 호출 생략]')
        return True
    rc, out, err = p18.run(cmd, timeout=timeout, env=p18.env_with(domain_id))
    if rc == 0:
        if 'response:' in out:
            snippet = out.split('response:')[-1].strip()
        else:
            snippet = out.strip()
        print(f'    ✓ 응답: {snippet[:200]}')
        return True
    print(f'    ✗ 실패 (rc={rc}): {(err or out).strip()[:300]}')
    return False


# ===== 동작들 =====
def show_status(domain_id, namespace, js_topic):
    print(f'\n=== 현재 관절 상태 ({js_topic}) — 한 메시지만 ===')
    rc, out, err = p18.run(
        ['ros2', 'topic', 'echo', js_topic, '--once'],
        timeout=8, env=p18.env_with(domain_id)
    )
    if rc == 0 and out.strip():
        print(out[:1200])
    else:
        print(f'  (못 받음) {err.strip()[:200]}')


def takeover(domain_id, namespace, dry_run=False):
    """충돌해제 → protective stop 해제 → 서보 ON → autonomous."""
    print('\n=== 통제권 가져오기 ===')
    seq = [
        ('1) 충돌(safe stop) 해제',
         'system/set_safe_stop_reset_type',
         'dsr_msgs2/srv/SetSafeStopResetType',
         '{reset_type: 1}'),
        ('2) Protective stop 해제',
         'system/release_protective_stop',
         'dsr_msgs2/srv/ReleaseProtectiveStop',
         '{}'),
        ('3) 서보 ON (STANDBY)',
         'system/set_robot_state',
         'dsr_msgs2/srv/SetRobotState',
         '{robot_state: 0}'),
        ('4) Autonomous 모드 (manual=0 / auto=1)',
         'system/set_robot_mode',
         'dsr_msgs2/srv/SetRobotMode',
         '{robot_mode: 1}'),
    ]
    ok_all = True
    for label, s, t, a in seq:
        print(f'\n {label}')
        ok = call_service(s, t, a, domain_id, namespace, dry_run=dry_run)
        ok_all = ok_all and ok
    print()
    if ok_all:
        print('  ✓ 통제권 확보 — 이제 motion 명령 가능')
    else:
        print('  ⚠ 일부 단계 실패 — 위 에러 메시지 확인')
    return ok_all


def movej(pos_list, domain_id, namespace, vel=30.0, acc=30.0, dry_run=False):
    if len(pos_list) != 6:
        print(f'  !! joint 값 6개 필요, 받은 개수: {len(pos_list)}')
        return False
    args_yaml = (f'{{pos: {list(pos_list)}, vel: {vel}, acc: {acc}, '
                 f'time: 0, radius: 0, mode: 0, blend_type: 0, sync_type: 0}}')
    print(f'\n=== movej {pos_list}  (vel={vel}, acc={acc}) ===')
    return call_service('motion/move_joint', 'dsr_msgs2/srv/MoveJoint',
                        args_yaml, domain_id, namespace,
                        dry_run=dry_run, timeout=30)


# ===== UI =====
def confirm(prompt, default=False):
    sfx = '[y/N]' if not default else '[Y/n]'
    try:
        ans = input(f'  ⚠ {prompt} {sfx}: ').strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not ans:
        return default
    return ans in ('y', 'yes')


def interactive_menu(domain_id, namespace, js_topic, dry_run=False):
    while True:
        print()
        print('=' * 64)
        flag = ' [DRY-RUN]' if dry_run else ''
        print(f'  메뉴 — domain={domain_id}, ns=/{namespace}{flag}')
        print('=' * 64)
        print('  [1] 현재 관절 상태 (read-only, 안전)')
        print('  [2] 통제권 가져오기 (충돌해제+서보ON+autonomous)')
        print('  [3] HOME 자세 (movej [0,0,90,0,90,0])')
        print('  [4] 사용자 입력 6개 관절각으로 movej')
        print('  [0] 종료')
        try:
            choice = input('선택: ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not choice or choice == '0':
            return
        if choice == '1':
            show_status(domain_id, namespace, js_topic)
        elif choice == '2':
            if dry_run or confirm('실제 로봇에 통제권 신호를 보냅니다. 진행?'):
                takeover(domain_id, namespace, dry_run=dry_run)
        elif choice == '3':
            if dry_run or confirm('HOME 으로 움직입니다 (vel=30, acc=30). 진행?'):
                movej([0, 0, 90, 0, 90, 0], domain_id, namespace, dry_run=dry_run)
        elif choice == '4':
            try:
                raw = input('  6개 관절각 (공백 구분, 예: 0 0 90 0 90 0): ').strip()
                pos = [float(x) for x in raw.split()]
            except (ValueError, EOFError, KeyboardInterrupt):
                print('  !! 숫자 형식 오류')
                continue
            if dry_run or confirm(f'movej {pos} 보냅니다. 진행?'):
                movej(pos, domain_id, namespace, dry_run=dry_run)
        else:
            print('  !! 잘못된 선택')


# ===== main =====
def main():
    ap = argparse.ArgumentParser(
        description='원격 두산 로봇 조작 — 18 스크립트 캐시 사용')
    ap.add_argument('--status', action='store_true', help='관절 상태 1회')
    ap.add_argument('--takeover', action='store_true', help='통제권 가져오기')
    ap.add_argument('--home', action='store_true', help='HOME movej')
    ap.add_argument('--movej', type=str, default=None,
                    help='"j1 j2 j3 j4 j5 j6" (공백 구분, 단위 deg)')
    ap.add_argument('--vel', type=float, default=30.0)
    ap.add_argument('--acc', type=float, default=30.0)
    ap.add_argument('--dry-run', action='store_true',
                    help='실제 호출 없이 명령만 출력')
    args = ap.parse_args()

    cache = p18.load_cache()
    if not cache:
        print('!! ~/.cache/doosan_remote.json 없음 — 먼저 18_원격_로봇_연결.py 실행')
        return 1

    domain_id = cache['domain_id']
    namespace = cache['namespace']
    status = cache.get('status', '?')
    js_topic = cache.get('joint_states_topic') or f'/{namespace}/joint_states'

    print('=' * 64)
    print(f'  원격 조작기  domain={domain_id}  ns=/{namespace}  '
          f'status={status}')
    print('=' * 64)
    if status != 'REAL':
        print(f'  ⚠ 현재 캐시 status={status} — REAL 환경 아닙니다.')
        print('     명령이 통하지 않을 수 있고, 18 스크립트로 REAL 잡기 권장')
    rips = cache.get('robot_ips', [])
    if rips:
        print(f'  로봇 IP: {", ".join(rips)}')

    os.environ['ROS_DOMAIN_ID'] = str(domain_id)
    os.environ.pop('ROS_LOCALHOST_ONLY', None)
    p18.stop_daemon()

    no_args = not any([args.status, args.takeover, args.home, args.movej])
    if no_args:
        return 0 if interactive_menu(domain_id, namespace, js_topic,
                                     dry_run=args.dry_run) is None else 0

    if args.status:
        show_status(domain_id, namespace, js_topic)
    if args.takeover:
        takeover(domain_id, namespace, dry_run=args.dry_run)
    if args.home:
        movej([0, 0, 90, 0, 90, 0], domain_id, namespace,
              vel=args.vel, acc=args.acc, dry_run=args.dry_run)
    if args.movej:
        try:
            pos = [float(x) for x in args.movej.split()]
        except ValueError:
            print('!! --movej 값 파싱 실패')
            return 1
        movej(pos, domain_id, namespace,
              vel=args.vel, acc=args.acc, dry_run=args.dry_run)
    return 0


if __name__ == '__main__':
    sys.exit(main())
