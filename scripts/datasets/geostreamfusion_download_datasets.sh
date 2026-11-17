#!/usr/bin/env bash
# GeoStreamFusion dataset downloader. Run this on 10.23.20.60.
#
#   ./geostreamfusion_download_datasets.sh --dry-run       show what would download, change nothing
#   ./geostreamfusion_download_datasets.sh                 download the required tier (307.0 GB)
#   ./geostreamfusion_download_datasets.sh --tier all      also fetch the optional sources (978.0 GB total)
#
# Options:
#   --root DIR           dataset root (default /home/vishnu/datasets, or $DATASET_ROOT)
#   --only REGEX         only files whose "group/path" matches, e.g. --only '^mars_lvig/AMvalley'
#   --jobs N             parallel downloads (default 3)
#   --ignore-extracted   re-download even when an extracted folder of the same name exists
#
# Safe to re-run: finished files are skipped and partial files resume. New files go under
# $ROOT/vnext_downloads/ and nothing already in $ROOT is modified. Archives are not extracted.
# Run it inside tmux or under nohup, since this takes many hours.
set -uo pipefail

ROOT="${DATASET_ROOT:-/home/vishnu/datasets}"
TIER=required ONLY='' JOBS=3 DRY=0 IGNORE_EXTRACTED=0
MIN_FREE_GB="${MIN_FREE_GB:-20}"     # space to leave free after downloading
UAVSCENES_REPO_COMMIT=909250d14a4460b2a62ea25f08f12ce6367427ac

die() { echo "ERROR: $*" >&2; exit 1; }
while [ $# -gt 0 ]; do
  case "$1" in
    --root) [ $# -ge 2 ] || die "--root needs a value"; ROOT=$2; shift 2 ;;
    --tier) [ $# -ge 2 ] || die "--tier needs a value"; TIER=$2; shift 2 ;;
    --only) [ $# -ge 2 ] || die "--only needs a value"; ONLY=$2; shift 2 ;;
    --jobs) [ $# -ge 2 ] || die "--jobs needs a value"; JOBS=$2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --ignore-extracted) IGNORE_EXTRACTED=1; shift ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done
case "$TIER" in required|all) ;; *) die "--tier must be required or all" ;; esac
case "$JOBS" in ''|*[!0-9]*|0) die "--jobs must be a positive integer" ;; esac
for c in wget awk sha256sum df find xargs tar flock; do
  command -v "$c" >/dev/null || die "missing command: $c"
done
command -v unzip >/dev/null || echo "WARNING: unzip not found, zip files will get a size check only"
[ -d "$ROOT" ] || die "dataset root does not exist: $ROOT"

DEST="$ROOT/vnext_downloads"
WORK="$DEST/_state"
[ "$DRY" = 1 ] && WORK=$(mktemp -d) || mkdir -p "$WORK" || die "cannot write to $DEST"
[ "$DRY" = 1 ] && trap 'rm -rf "$WORK"' EXIT
if [ "$DRY" = 0 ]; then
  # Two copies appending to one partial file would corrupt it. Child wgets inherit this lock,
  # so it also holds while downloads from a killed run are still finishing.
  exec 9>"$WORK/lock"
  flock -n 9 || die "another copy is still running on $ROOT (check: pgrep -af vnext_downloads)"
fi
QUEUE="$WORK/queue.tsv" STATUS="$WORK/status.tsv" EXISTING="$WORK/existing.tsv"
SCRIPT_MANIFEST() { sed -n '/^#__MANIFEST__$/,$p' "$0" | tail -n +2; }

url_for() {  # gd:<id> or hf:<repo>@<commit>:<path>
  case "$1" in
    gd:*) echo "https://drive.usercontent.google.com/download?id=${1#gd:}&export=download&confirm=t" ;;
    hf:*) local s repo rest
          s=${1#hf:}; repo=${s%%@*}; rest=${s#*@}
          echo "https://huggingface.co/datasets/$repo/resolve/${rest%%:*}/${rest#*:}" ;;
  esac
}

# Asks the host for one byte. The total in Content-Range must equal the manifest size, so a
# Google Drive quota page or a replaced file is caught before anything is written.
remote_size() {
  wget -nv -S --tries=2 --timeout=60 --header='Range: bytes=0-0' -O /dev/null "$1" 2>&1 | tr -d '\r' |
    awk 'tolower($1)=="content-range:"{n=split($3,a,"/"); s=a[n]} END{print s}'
}

fetch_one() {
  local tier group rel bytes sha src
  IFS=$'\t' read -r tier group rel bytes sha src <<<"$1"
  local dst="$DEST/$group/$rel" part="$DEST/$group/$rel.part" url have try rc
  url=$(url_for "$src")
  mkdir -p "$(dirname "$dst")"
  note() { printf '%s\t%s/%s\t%s\n' "$1" "$group" "$rel" "${2:-}" >>"$STATUS"; echo "[$1] $group/$rel ${2:-}"; }

  if [ -f "$dst" ] && [ "$(stat -c %s "$dst")" = "$bytes" ]; then note OK "already complete"; return 0; fi
  if [ "$(remote_size "$url")" != "$bytes" ]; then
    note BLOCKED "host did not offer the expected size (Drive quota, or the file changed); re-run later"
    return 0
  fi
  for try in 1 2 3 4 5 6; do
    # A host error page or an oversized file must not be resumed, or new bytes get appended to it.
    have=$(stat -c %s "$part" 2>/dev/null || echo 0)
    if [ "$have" -gt "$bytes" ] ||
       { [ "$have" -lt 1048576 ] && head -c 64 "$part" 2>/dev/null | grep -qiE '<(!doctype|html)'; }; then
      rm -f "$part"
      have=0
    fi
    [ "$have" = "$bytes" ] && break
    wget -nv --continue --tries=1 --timeout=120 -O "$part" "$url"
    rc=$?
    have=$(stat -c %s "$part" 2>/dev/null || echo 0)
    [ "$have" = "$bytes" ] && break
    echo "  retry $try for $rel (wget exit $rc, $have of $bytes bytes)"
    sleep $((try * ${RETRY_DELAY:-30}))
  done
  have=$(stat -c %s "$part" 2>/dev/null || echo 0)
  if [ "$have" != "$bytes" ]; then note FAILED "got $have of $bytes bytes; re-run to resume"; return 0; fi
  if [ "$sha" != "-" ] && [ "$(sha256sum "$part" | cut -d' ' -f1)" != "$sha" ]; then
    mv -f "$part" "$part.badsha"; note FAILED "SHA-256 mismatch, kept as .badsha"; return 0
  fi
  case "$rel" in *.zip)
    if command -v unzip >/dev/null && ! nice -n 10 unzip -tqq "$part" >/dev/null 2>&1; then
      mv -f "$part" "$part.badzip"; note FAILED "zip CRC test failed, kept as .badzip"; return 0
    fi ;;
  esac
  mv -f "$part" "$dst"
  local checks="size only"
  [[ "$rel" == *.zip ]] && checks="size+zip-crc"
  [ "$sha" != "-" ] && checks="size+sha256"
  note OK "$checks"
}
export -f fetch_one url_for remote_size
export DEST STATUS

# Index what the server already holds, leaving out our own folder.
find "$ROOT" -maxdepth 5 -path "$DEST" -prune -o \
  \( -name 'interval1_*' -o -name '*.bag' -o -name 'terra_3dmap_pointcloud_mesh*' \) \
  -printf '%y\t%s\t%f\t%p\n' 2>/dev/null >"$EXISTING"

: >"$QUEUE"
SKIPPED=0 SKIPPED_BYTES=0 TODO_BYTES=0 EXPECTED=0
while IFS=$'\t' read -r tier group rel bytes sha src; do
  [ "$TIER" = all ] || [ "$tier" = required ] || continue
  [ -z "$ONLY" ] || [[ "$group/$rel" =~ $ONLY ]] || continue
  name=${rel##*/} stem=${name%.*} found=''
  case "$group" in uavscenes|mars_lvig)
    found=$(awk -F'\t' -v n="$name" -v b="$bytes" '$1=="f" && $3==n && $2==b {print $4; exit}' "$EXISTING")
    if [ -z "$found" ] && [ "$IGNORE_EXTRACTED" = 0 ]; then
      found=$(awk -F'\t' -v n="$stem" '$1=="d" && $3==n {print $4; exit}' "$EXISTING")
      [ -z "$found" ] || found="$found (extracted folder, completeness not checked)"
    fi ;;
  esac
  if [ -n "$found" ]; then
    echo "skip $group/$rel: already at $found"
    SKIPPED=$((SKIPPED + 1)) SKIPPED_BYTES=$((SKIPPED_BYTES + bytes))
    continue
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$tier" "$group" "$rel" "$bytes" "$sha" "$src" >>"$QUEUE"
  EXPECTED=$((EXPECTED + 1))
  if [ -f "$DEST/$group/$rel" ] && [ "$(stat -c %s "$DEST/$group/$rel")" = "$bytes" ]; then continue; fi
  partial=$(stat -c %s "$DEST/$group/$rel.part" 2>/dev/null || echo 0)
  TODO_BYTES=$((TODO_BYTES + bytes - partial))
done < <(SCRIPT_MANIFEST)

gb() { awk -v b="$1" 'BEGIN{printf "%.1f", b/1e9}'; }
echo
echo "Tier: $TIER${ONLY:+ (filter: $ONLY)}   Root: $ROOT"
awk -F'\t' '{f[$2]++; s[$2]+=$4} END{for(g in f) printf "  %-20s %4d files %9.1f GB\n", g, f[g], s[g]/1e9}' "$QUEUE" | sort
echo "  Skipped as already present: $SKIPPED files, $(gb "$SKIPPED_BYTES") GB"
echo "  Still to download:          $(gb "$TODO_BYTES") GB"
FREE=$(df -B1 --output=avail "$ROOT" | tail -1 | tr -d ' ')
echo "  Free space on $ROOT: $(gb "$FREE") GB (keeping $MIN_FREE_GB GB free)"
ENOUGH=1
awk -v t="$TODO_BYTES" -v f="$FREE" -v r="$MIN_FREE_GB" 'BEGIN{exit !(t + r*1e9 <= f)}' || ENOUGH=0
if [ "$ENOUGH" = 0 ]; then
  echo "NOT ENOUGH DISK SPACE."
  [ "$DRY" = 1 ] && exit 0
  die "free up space, or narrow the selection with --only"
fi
[ "$DRY" = 1 ] && { echo "Dry run: nothing downloaded."; exit 0; }

for host in https://huggingface.co https://drive.usercontent.google.com; do
  wget -q --spider --tries=1 --timeout=20 "$host"
  rc=$?
  # 8 only means the host answered with an HTTP error, which still proves it is reachable.
  [ "$rc" = 0 ] || [ "$rc" = 8 ] || die "cannot reach $host from this machine (wget exit $rc)"
done

if [[ -z "$ONLY" || "uavscenes/helper_repo" =~ $ONLY ]] && [ ! -d "$DEST/uavscenes/helper_repo" ]; then
  mkdir -p "$DEST/uavscenes/helper_repo.tmp"
  # Holds the camera-LiDAR calibration files that Stage 1 needs.
  wget -q -O - "https://codeload.github.com/sijieaaa/UAVScenes/tar.gz/$UAVSCENES_REPO_COMMIT" |
    tar -xz --strip-components=1 -C "$DEST/uavscenes/helper_repo.tmp" &&
    mv "$DEST/uavscenes/helper_repo.tmp" "$DEST/uavscenes/helper_repo" ||
    echo "WARNING: could not fetch the UAVScenes helper repo; re-run to retry"
fi

cat >"$DEST/LICENCE_REGISTER.tsv" <<REG
dataset	source	version	accessed	terms	intended_use
uavscenes	https://github.com/sijieaaa/UAVScenes	repo $UAVSCENES_REPO_COMMIT	$(date +%F)	README: CC BY-NC-SA 4.0, academic only; GitHub licence metadata is unset, confirm before release	primary measured-depth development and utility labels
mars_lvig	https://mars.hku.hk/dataset.html	Drive update folder 1aG21le4QZl9LhSLE1O0vuVDQ8YgAwdas	$(date +%F)	CC BY-NC-SA 4.0, academic only	raw timing, IMU and deskew investigation
mun_frl	https://mun-frl-vil-dataset.readthedocs.io/	Drive folders	$(date +%F)	CC BY 4.0, credit the paper	candidate independent tracking and geometry test
ntu_viral	https://ntu-aris.github.io/ntu_viral_dataset	HF 4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd	$(date +%F)	CC BY-NC-SA 4.0 on site, CC BY-NC 4.0 on DR-NTU; confirm before release	candidate independent tracking test
matrixcity_block3	https://huggingface.co/datasets/BoDai/MatrixCity	HF 22237509a7a16d5c0136b58b39597629a63b338d	$(date +%F)	CC BY-NC 4.0	bounded exact-truth projection and evaluator check
REG

echo "Starting at $(date), $JOBS parallel downloads. Progress lines appear as each file finishes."
: >"$STATUS"
xargs -a "$QUEUE" -d '\n' -P "$JOBS" -I LINE bash -c 'fetch_one "$1"' _ LINE

OK=$(grep -c '^OK' "$STATUS"); BAD=$(grep -c -v '^OK' "$STATUS")
echo
echo "Finished at $(date): $OK of $EXPECTED files verified, $BAD not complete."
if [ "$BAD" -gt 0 ] || [ "$OK" != "$EXPECTED" ]; then
  grep -v '^OK' "$STATUS"
  echo "Re-run the same command to resume. If Google Drive reports BLOCKED, wait 24 hours first."
  exit 2
fi
cat <<'NOTE'
All selected files are verified. Two sources cannot be automated and were not downloaded:
  UseGeo:  open these in a browser, https://eostore.itc.utwente.nl:5001/sharing/1gJRLdQ71
           .../sharing/c4LlTkVjT and .../sharing/r4o1tdCNv
  Mid-Air: needs an email request at https://midair.ulg.ac.be/download.html
NOTE
exit 0
#__MANIFEST__
all	uavscenes	interval5_CAM_label.zip	1488319248	461909f53b45933824597226cc5a7de70c04793c38f1db73056eabbf525c9fd5	hf:sijieaaa/UAVScenes@04a224fd14b0b8fdfd2633e68f20471b85c611a5:interval5_CAM_label.zip
all	uavscenes	interval5_LIDAR_label.zip	105289464	14e5fbc5d775887a6ac4ca135890a9903327c01a7f90901a2b4b3a0bd9239942	hf:sijieaaa/UAVScenes@04a224fd14b0b8fdfd2633e68f20471b85c611a5:interval5_LIDAR_label.zip
required	uavscenes	interval1_CAM_LIDAR/interval1_AMtown01.zip	15217419198	-	gd:1RBU4Aj4F3beFAh8o1Gmjr7dDbTSbLdXf
required	uavscenes	interval1_CAM_LIDAR/interval1_AMtown02.zip	8135415915	-	gd:1Dk9q5n3FzqkQu0v9llN4DVkYJ1RfHn04
required	uavscenes	interval1_CAM_LIDAR/interval1_AMtown03.zip	6492106212	-	gd:12BYqamvO9gxrH_CxBRft4imQ12-EucjP
required	uavscenes	interval1_CAM_LIDAR/interval1_AMvalley01.zip	12274794304	-	gd:1RTIVChLYu9GvSBuT_Pzxe703Gm1t_jO5
required	uavscenes	interval1_CAM_LIDAR/interval1_AMvalley02.zip	7122124537	-	gd:1uYIzexNMRuNEiJ-PG57j9V6jbKxqs-Vw
required	uavscenes	interval1_CAM_LIDAR/interval1_AMvalley03.zip	5290486076	-	gd:1jFhXnBG50CMQLKQGAX2sTrbuBRgqK4RD
required	uavscenes	interval1_CAM_LIDAR/interval1_HKairport01.zip	9898021677	-	gd:1XNUFDYVoIRWNS2_-hV3jIlgk1ciyfxUM
required	uavscenes	interval1_CAM_LIDAR/interval1_HKairport02.zip	5214262577	-	gd:1qizeyDUCX73r1XsNYVSDX7Bbh4Hzywod
required	uavscenes	interval1_CAM_LIDAR/interval1_HKairport03.zip	3979883447	-	gd:1E-zaRQlemh4rnmXUd5y63mRdpVH3ngWF
required	uavscenes	interval1_CAM_LIDAR/interval1_HKisland01.zip	8088227154	-	gd:1fI9jgo84REFWWMAxe--7hY1nKhWP6UDm
required	uavscenes	interval1_CAM_LIDAR/interval1_HKisland02.zip	4875325897	-	gd:1YZLZONXzSbXcY3X0GS__XF66bAoGyUqr
required	uavscenes	interval1_CAM_LIDAR/interval1_HKisland03.zip	3769048048	-	gd:1do6WzVc7_V1CS9rqMi96iUXEu5c5WLRq
required	uavscenes	interval1_CAM_label.zip	7440189777	-	gd:1JM_BgNJDBPemJ75XAA77RkFV2pmEoM5i
required	uavscenes	interval1_LIDAR_label.zip	526185143	-	gd:1AAKgR4dmbJiNhO5PXOLfdh5KEHI_xqgS
required	uavscenes	terra_3dmap_pointcloud_mesh.zip	6779103543	-	gd:1ol_Ngk0XwMlDIYvk2BqgHna3w2KPB0dc
required	mars_lvig	AMtown01.bag	32085746803	-	gd:1mXEWG8sxR0V0JHaSD-upAgK21Ge6ybNa
required	mars_lvig	AMtown02.bag	17706120943	-	gd:15bio33_AMrumJM-zZCzvdo9P1JBirhcQ
required	mars_lvig	AMtown03.bag	14515362776	-	gd:1m6-Um5Jd3mLRKHpBlKgGbtoV-kPv4eKa
required	mars_lvig	AMvalley01.bag	27711887094	-	gd:1NTecR3tb2-NYZDPH_p94bFy3lYmsQ53b
required	mars_lvig	AMvalley02.bag	16616088448	-	gd:1g5YQ6Po0gF_HhcJ24A03AJOOrybmSjOJ
required	mars_lvig	AMvalley03.bag	12844627112	-	gd:18B2yu7HHPHd-Wy3HHgAC74_5FsgBK7F3
required	mars_lvig	HKairport01.bag	19900272121	-	gd:1dj_7Htdb1rtRsiQ-HqcndrBxV4GOr_2H
required	mars_lvig	HKairport02.bag	11359145914	-	gd:1hxSdFyULhS2MlVG5Aba9YWFc9DambsuA
required	mars_lvig	HKairport03.bag	9080929458	-	gd:1V2Z8eUuVOanpdZiga7LcskgzfJD3gEqn
required	mars_lvig	HKisland01.bag	18839130084	-	gd:1nhX7hGyjCaoIfqc2b3PhAaHHu2Vv8wOQ
required	mars_lvig	HKisland02.bag	11787624886	-	gd:1CPOE1y6fVz20f4tbFv44TWMatihb8JTZ
required	mars_lvig	HKisland03.bag	9459767048	-	gd:1Now3Dz8UIHkwya8YvoJYHzukEWEBAobB
all	uavscenes	interval1_CAM_LIDAR/interval1_HKairport_GNSS01.zip	9136952219	-	gd:1tLqz-OH8weS2jtBIi6PaCWhYhVjL1N8E
all	uavscenes	interval1_CAM_LIDAR/interval1_HKairport_GNSS02.zip	4950782732	-	gd:1DzMnKjJcmm1GvbTqjbGHa8UrC0nAVVRS
all	uavscenes	interval1_CAM_LIDAR/interval1_HKairport_GNSS03.zip	3984846544	-	gd:1HbXOp7e17yh1ulyUsQSWMpfs-ITGobAV
all	uavscenes	interval1_CAM_LIDAR/interval1_HKairport_GNSS_Evening.zip	7497016302	-	gd:1V-rtMwZIiOi116BagYyOcnVXXLw8dn7u
all	uavscenes	interval1_CAM_LIDAR/interval1_HKisland_GNSS01.zip	8886699478	-	gd:1yRfAYRK9mL9vzdvCysik_7O8G_Fx3Zz7
all	uavscenes	interval1_CAM_LIDAR/interval1_HKisland_GNSS02.zip	5038068809	-	gd:1yRRo3omXMihmsEuWd1JKnJwYdurpgggQ
all	uavscenes	interval1_CAM_LIDAR/interval1_HKisland_GNSS03.zip	3479676247	-	gd:1_BVH9-4ID-FStEIWfqBLUVA_IxyS6bb9
all	uavscenes	interval1_CAM_LIDAR/interval1_HKisland_GNSS_Evening.zip	9866654153	-	gd:1QlJvooQGpE0EOpEy0HWuUIyuZXixvpoG
all	mars_lvig	HKairport_GNSS01.bag	20181345330	-	gd:1F61oziWPbeaqP5fFtuiqvl_3BvrvWwz0
all	mars_lvig	HKairport_GNSS02.bag	11637475630	-	gd:1OVFhhpkFReSy7j1AYIYQG4rbWL_ZRP-3
all	mars_lvig	HKairport_GNSS03.bag	9756614351	-	gd:1ORuC3xyi7yonIzD8mTmKDWRU0T6snu9n
all	mars_lvig	HKisland_GNSS01.bag	20110103618	-	gd:1YG22xjU6q-AECsizsrshgfgq0lUWDe7j
all	mars_lvig	HKisland_GNSS02.bag	12085085081	-	gd:1UDcZgElsNepsJXgE9Wx1djHrLNiNwqyP
all	mars_lvig	HKisland_GNSS03.bag	10040751440	-	gd:136xN27vdpUe88NugHhPKArZBveiYD3hj
all	mun_frl	quarry1/flight_dataset2.bag	29198599879	-	gd:1_pN_kRbYmPEgHbCybr3uG9bV7LMtG_HC
all	mun_frl	quarry1/flight_dataset2.ubx	2607136	-	gd:1Rc5jSbEWBm_q_fupoM7T7K3p2nq9tt5e
all	mun_frl	quarry1/ppk_data/quarry_1.pdf	6316096	-	gd:1MMhgyDaDR1QFDVqosnUpLS87KV6rtffx
all	mun_frl	quarry1/ppk_data/quarry_1.pos	164232	-	gd:1Gpr5JKE5iNiB0H5RXA4Kpb78nkKUP2oN
all	mun_frl	quarry1/ppk_data/quarry_1_ppk.kml	145640	-	gd:1STI3rOm5944ukhxghEKX4j9ZOHgVqJj1
all	mun_frl	quarry1/ppk_data/quarry_1_ppk.pos	123660	-	gd:1_o1gtOxE9OKpIxHf8rQa-njHIaFDRnmz
all	mun_frl	quarry1/ppk_data/reach_base_raw_202202221911.UBX	3575256	-	gd:1cbu6TwFWEy165haAhGTvXf0J9K1M8ZYK
all	mun_frl	quarry1/flight_log/quarry1.csv	422519	-	gd:1QRnhTLUtrZkLARAFse2oGK77ewEK3s2p
all	mun_frl	quarry2/flight_dataset3.bag	111720467155	-	gd:1aysbe99efDsrR5pYCvV-upV9ylzFT8Nm
all	mun_frl	quarry2/flight_dataset3.ubx	8806696	-	gd:1Q_Pmc5YCrtbKteVfksnc0i9TWvdBUUGJ
all	mun_frl	quarry2/ppk_data/quarry_2.pdf	8072262	-	gd:1L_5omI-pnF5LAK8tEwSH-H42-HlHQDrR
all	mun_frl	quarry2/ppk_data/quarry_2.pos	630560	-	gd:1pMcjf1dgpk15WkEaW6NASxB-0XKCWScL
all	mun_frl	quarry2/ppk_data/quarry_2_ppk.kml	742331	-	gd:1mfTjPCswmU-widwzB6LzpxvJmfd14d9c
all	mun_frl	quarry2/ppk_data/quarry_2_ppk.pos	631026	-	gd:1OH3xXRmlJZkQfkF8ayqkwyFddo-FJcTm
all	mun_frl	quarry2/ppk_data/reach_base_raw_202202252019.UBX	49343584	-	gd:1Yb5e5kFXqp_h9Aq4zMzFpB0wbZRLroSj
all	mun_frl	quarry2/flight_log/quarry2.jpg	424579	-	gd:1Zw7Yo4fnpNBTMCe3z5oT1kK-M31qA5tg
all	mun_frl	lighthouse/flight_dataset5.ubx	8982528	-	gd:1fNQPFTpT8WjH1IdaBf8STwyAvHCbTaf7
all	mun_frl	lighthouse/flight_dataset5_fixed.bag	95682967831	-	gd:1gmWHDp2VBVTOhuJaDJK2HVm3KeoAR5d0
all	mun_frl	lighthouse/ppk_data/flight_dataset5.pos	546520	-	gd:1EGOvh3iszz7aJPxXviq1KbPPaJH4Xv17
all	mun_frl	lighthouse/ppk_data/flight_dataset5_ppk.kml	643467	-	gd:14Xp3sz7fRz8YXUABK8cMQJQk6kxM_nfc
all	mun_frl	lighthouse/ppk_data/flight_dataset5_ppk.pos	546968	-	gd:1G88Zn8D89oLvV_ExBxOtbgLTpJT9zU6x
all	mun_frl	lighthouse/ppk_data/light_house.pdf	12733151	-	gd:1EOtsxVqCkuyWqqh58XfMtNFNWdyRTFfH
all	mun_frl	lighthouse/ppk_data/reach_base_raw_202202282015.UBX	37944652	-	gd:1mIX_GfT3578NoBfQWoxrcDiQk2kIEaOZ
all	mun_frl	lighthouse/map_pcd/lighthouse_map.pcd	55238760	-	gd:1uewcGnqdFz-3R4P6UDflKCoA6d9qVhz9
all	mun_frl	lighthouse/flight_log/lighthouse.csv	2863730	-	gd:12G847tFKQGt9O2JUFwq5r9_qfw-PO-5w
all	mun_frl	bell412_dataset1/bell412_dataset1.bag	49288283371	-	gd:1VYdTzzTQgPXgkK0t-gt0qdzPPuRpVfKZ
all	mun_frl	bell412_dataset1/bell412_dataset1.ubx	5421712	-	gd:1tvPczonx9E6-TssckcP_SydQV0GSjcoe
all	mun_frl	bell412_dataset1/dataset1_2_21_pxpm_map_parameters_0.453515_-429_-612.jpg	8911184	-	gd:1LFNMOu6GMj0CR3mArrZ-ZCQ5tFsO-2Dc
all	mun_frl	bell412_dataset1/ppk_data/bell412_dataset1.pos	320906	-	gd:1PYH0gJcyOyBVqpH4EoVCNJekPFEcnDcl
all	mun_frl	bell412_dataset1/ppk_data/bell412_dataset1_frl.pos	475421	-	gd:14blQp61TXsJWpynAyUqd5QwHjYCvMp7I
all	mun_frl	bell412_dataset1/ppk_data/bell412_dataset1_ppk.kml	378104	-	gd:1opi0B-Fb_QWaojA-5WCCZ9NbWivq_Lr6
all	mun_frl	bell412_dataset1/ppk_data/bell412_dataset1_ppk.pos	321372	-	gd:1zK2UMIZUWkcNl0k01pwq4I5xa0vSvGYW
all	mun_frl	bell412_dataset1/ppk_data/dataset1_length_145m.pdf	12145879	-	gd:10ybuuHECWsQAXCY8d1b2m9bmoMuYRTvj
all	mun_frl	bell412_dataset1/ppk_data/GPSPositionsBell1_time.txt	94613	-	gd:1BAV-uxy3VuppMjTPPR8L8wMQio8PVP4b
all	mun_frl	bell412_dataset1/ppk_data/Screen_Shot_2022-08-22_at_9.50.48_AM.png	1271835	-	gd:1KxsvgJ8R2VWqD3jvBFg4RbHLTSPN6Mih
all	mun_frl	bell412_dataset1/ppk_data/VINSPositionsBell1_time.txt	372393	-	gd:12YIKDnlIE1KRsK0pX0qKtyEnmnnz6i8Z
all	mun_frl	bell412_dataset1/map_pcd/bell412_dataset1.pcd	58725520	-	gd:133qtlNEIImY31WqaUsLOFV4N73f6ini4
all	mun_frl	bell412_dataset2/bell412_dataset2.ubx	5309860	-	gd:1kqnqD8UUPFuyZfWNSMy9DuMwloxUwNE6
all	mun_frl	bell412_dataset2/dataset2_0_734_pxpm_map_yaml_1_362398_-3784_-2922.jpg	8922582	-	gd:1y9llS-aaX7s9g9kgecZIKsVSVJ-OJwaN
all	mun_frl	bell412_dataset2/ppk_data/bell412_dataset2.kml	293602	-	gd:1nD6SnkPsrdommip_WYhmqcdJ2B3lx2cB
all	mun_frl	bell412_dataset2/ppk_data/bell412_dataset2.pos	329000	-	gd:1FTK5QWoe4-9cQbEqaqR2VfrKqu8FkyN6
all	mun_frl	bell412_dataset2/ppk_data/bell412_dataset2_frl.pos	491219	-	gd:13wTOElgrKPxdtf6M2SJ_nuP_8A7kv6iH
all	mun_frl	bell412_dataset2/ppk_data/bell412_dataset2_ppk.pos	249509	-	gd:1It0Qbp2U7jPTEbw4pYyzgnkAo1JnMZ5s
all	mun_frl	bell412_dataset3/bell412_dataset3.bag	34521059625	-	gd:1AUbBQNjY3xyBXSbsn83p88JsQI-JXASo
all	mun_frl	bell412_dataset3/bell412_dataset3.ubx	3336496	-	gd:1B3bGhyRBQD0CYYiP-ktDTLcZyVHx-Zc8
all	mun_frl	bell412_dataset3/dataset32_1_26_pxpm_map_parameters_0_793651_-1788_-1465.jpg	8561789	-	gd:1g2ABz8NvGWX-AsetDxKnSfQTGzSXodCW
all	mun_frl	bell412_dataset3/ppk_data/bell412_dataset3.pos	219660	-	gd:1BUYX8DZ39VWquv3F_pScmUwC7vbPV_nB
all	mun_frl	bell412_dataset3/ppk_data/bell412_dataset3_frl.pos	328615	-	gd:1EGQTdouR1ZCMWMFeye4szolx5tFaW4ql
all	mun_frl	bell412_dataset3/ppk_data/bell412_dataset3_ppk.kml	259033	-	gd:1en2AiKm0IPgoOT_Au2FQbpcbFnkDh4n3
all	mun_frl	bell412_dataset3/ppk_data/bell412_dataset3_ppk.pos	220126	-	gd:1GvOMj341y_dITWkWXnQi7lQ5o4YzpCzm
all	mun_frl	bell412_dataset3/map_pcd/bell412_dataset3.pcd	35924800	-	gd:1Dc6Tfa3cv7SIHy1Ee0XqGOswXXVvwMvG
all	mun_frl	bell412_dataset4/bell412_dataset4.bag	35471681295	-	gd:1bUhK2grHwcMlBa7AS_7u_59OrkzSTMN_
all	mun_frl	bell412_dataset4/bell412_dataset4.ubx	3403772	-	gd:1TT3tctysU8n0OjJPoKZ-2QHsBMWFrnD9
all	mun_frl	bell412_dataset4/dataset42_1_298_pxpm_map_parameters_0_770416_-1877_-61.jpg	8158841	-	gd:1HG_wm3xC5kStmiKNzqcsDDa4z9feCmSd
all	mun_frl	bell412_dataset4/ppk_data/bell412_dataset4.pos	227470	-	gd:1G830QfG-A2giAOgkyRFH0SCOYH-qX9Vh
all	mun_frl	bell412_dataset4/ppk_data/bell412_dataset4_frl.pos	338449	-	gd:1nnZqsSc12RyQp6HlzIjOQLzxXyVsAeHm
all	mun_frl	bell412_dataset4/ppk_data/bell412_dataset4_ppk.kml	268218	-	gd:1goDcGfVkAO2n83_IkrraoPqHZxnfU0zz
all	mun_frl	bell412_dataset4/ppk_data/bell412_dataset4_ppk.pos	227936	-	gd:1bUc83cJ5DeQRI09DOQLwUi4vnFNp1hUv
all	mun_frl	bell412_dataset4/ppk_data/dataset5.pdf	13675864	-	gd:1KmfdUrEE5WJewOf3dStfZaQ8jVm1LGEg
all	mun_frl	bell412_dataset4/map_pcd/bell412_dataset4.pcd	40802720	-	gd:13raLcssd5Ff1yz17kyu6hq6Ze57OMyEb
all	mun_frl	bell412_dataset5/bell412_dataset5.bag	36579164394	-	gd:1ZoGYN5GTT6nOw_Bz4X3Ef5zaof7W746c
all	mun_frl	bell412_dataset5/bell412_dataset5.ubx	5571184	-	gd:1V1zsAm9EGFFGqTFl3kvFR4wklhI5s-8E
all	mun_frl	bell412_dataset5/dataset5_2_001_pxpm_map_yaml_0_499750_-1009_-441.jpg	8599109	-	gd:13l8IToWtAf08VWDg5VrA4t7GATGHbhnW
all	mun_frl	bell412_dataset5/ppk_data/bell412_dataset5.pos	344052	-	gd:1R4cWsUGuWMfcfG5tCf4reLxA6O0Ayxfi
all	mun_frl	bell412_dataset5/ppk_data/bell412_dataset5_frl.pos	513551	-	gd:1vMNRDUTxcJz-5e49FwJB5ij-kuFfYU_Z
all	mun_frl	bell412_dataset5/ppk_data/bell412_dataset5_ppk.kml	405325	-	gd:10Tf8j4HqGKMxKALShfAmWNlNu2Povd78
all	mun_frl	bell412_dataset5/ppk_data/bell412_dataset5_ppk.pos	344518	-	gd:1UE4eAAWE71MqgqdFbR0fWDRxuNlMCL5w
all	mun_frl	bell412_dataset5/ppk_data/GPSPositionsBell5_time.txt	75175	-	gd:1-3xAdGLg3ViuikkjAQLl0RN9gpjPzx-Q
all	mun_frl	bell412_dataset5/ppk_data/VINSPositionsBell5_time.txt	291359	-	gd:1444qtcyrBqYsacJ9FYkSKYyWvnfubNVL
all	mun_frl	bell412_dataset5/map_pcd/bell412_dataset5.pcd	39424280	-	gd:1ms8wnFXObXI-0SKQEgDuqpKWH29dowo2
all	mun_frl	bell412_dataset6/bell412_dataset6.bag	51086320202	-	gd:1Rc9PD15mWIlOJZlmg9nZ0TiH3fzDYT0M
all	mun_frl	bell412_dataset6/dataset6_3_602_pxpm_map_yaml_parameters_0_277624_-502_-86.jpg	8074716	-	gd:1SE3dJz3qe5T0tWwtT33NgPPTCJCO0uSU
all	mun_frl	bell412_dataset6/ppk_data/Bell412_dataset6.kml	96274	-	gd:1tVL6bH-69Wzl28JslunFZzgK4VQkBzd7
all	mun_frl	bell412_dataset6/ppk_data/bell412_dataset6_frl.pos	552355	-	gd:1sE5cfKJ4DHxjXvHAxtj_M34b6Zjso1GW
all	mun_frl	bell412_dataset6/map_pcd/bell412_dataset6.pcd	65293120	-	gd:1B-B-k8IZJ3oYzFlqSRduvwq7VGaKKvvX
all	ntu_viral	BASELINES.md	77318	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:BASELINES.md
all	ntu_viral	COMMON_ISSUES.md	1855	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:COMMON_ISSUES.md
all	ntu_viral	EVALUATION.md	2397	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:EVALUATION.md
all	ntu_viral	NTU_VIRAL.bib	409	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:NTU_VIRAL.bib
all	ntu_viral	PROVENANCE.md	3250	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:PROVENANCE.md
all	ntu_viral	README.md	41710	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:README.md
all	ntu_viral	REFERENCES.bib	41511	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:REFERENCES.bib
all	ntu_viral	RELEASE_VERIFICATION.json	96522	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:RELEASE_VERIFICATION.json
all	ntu_viral	SENSORS_AND_CALIBRATION.md	4047	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:SENSORS_AND_CALIBRATION.md
all	ntu_viral	SHA256SUMS.txt	7148	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:SHA256SUMS.txt
all	ntu_viral	calibration/calib_stereo.zip	51118082	8ac158143ddb42f86f73cb064128621a49cf30b67d411f2586cc04ad959ceb0f	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:calibration/calib_stereo.zip
all	ntu_viral	calibration/calib_stereo_imu.bag	1005659191	ad3e50c52e72683e4967271d94c20bf5ae9096fdbb432be1044dc14cec89f8f3	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:calibration/calib_stereo_imu.bag
all	ntu_viral	ground_truth/eee_01/ground_truth.csv	782140	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:ground_truth/eee_01/ground_truth.csv
all	ntu_viral	ground_truth/eee_02/ground_truth.csv	654026	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:ground_truth/eee_02/ground_truth.csv
all	ntu_viral	ground_truth/eee_03/ground_truth.csv	352902	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:ground_truth/eee_03/ground_truth.csv
all	ntu_viral	ground_truth/nya_01/ground_truth.csv	928153	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:ground_truth/nya_01/ground_truth.csv
all	ntu_viral	ground_truth/nya_02/ground_truth.csv	907988	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:ground_truth/nya_02/ground_truth.csv
all	ntu_viral	ground_truth/nya_03/ground_truth.csv	926845	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:ground_truth/nya_03/ground_truth.csv
all	ntu_viral	ground_truth/rtp_01/ground_truth.csv	1951124	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:ground_truth/rtp_01/ground_truth.csv
all	ntu_viral	ground_truth/rtp_02/ground_truth.csv	647922	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:ground_truth/rtp_02/ground_truth.csv
all	ntu_viral	ground_truth/rtp_03/ground_truth.csv	760246	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:ground_truth/rtp_03/ground_truth.csv
all	ntu_viral	ground_truth/sbs_01/ground_truth.csv	663282	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:ground_truth/sbs_01/ground_truth.csv
all	ntu_viral	ground_truth/sbs_02/ground_truth.csv	714725	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:ground_truth/sbs_02/ground_truth.csv
all	ntu_viral	ground_truth/sbs_03/ground_truth.csv	640432	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:ground_truth/sbs_03/ground_truth.csv
all	ntu_viral	ground_truth/spms_01/ground_truth.csv	809405	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:ground_truth/spms_01/ground_truth.csv
all	ntu_viral	ground_truth/spms_02/ground_truth.csv	658001	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:ground_truth/spms_02/ground_truth.csv
all	ntu_viral	ground_truth/spms_03/ground_truth.csv	477725	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:ground_truth/spms_03/ground_truth.csv
all	ntu_viral	ground_truth/tnp_01/ground_truth.csv	1101627	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:ground_truth/tnp_01/ground_truth.csv
all	ntu_viral	ground_truth/tnp_02/ground_truth.csv	910344	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:ground_truth/tnp_02/ground_truth.csv
all	ntu_viral	ground_truth/tnp_03/ground_truth.csv	797092	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:ground_truth/tnp_03/ground_truth.csv
all	ntu_viral	metadata/BASELINE_CITATIONS.json	41944	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:metadata/BASELINE_CITATIONS.json
all	ntu_viral	metadata/BASELINE_SEARCH_AUDIT.json	16361	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:metadata/BASELINE_SEARCH_AUDIT.json
all	ntu_viral	metadata/BASELINE_SOURCES.json	110236	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:metadata/BASELINE_SOURCES.json
all	ntu_viral	metadata/GITHUB_SOURCE_MANIFEST.json	21415	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:metadata/GITHUB_SOURCE_MANIFEST.json
all	ntu_viral	metadata/NTU_DATAVERSE_V5.json	34262	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:metadata/NTU_DATAVERSE_V5.json
all	ntu_viral	metadata/SOURCE_MANIFEST.json	5891	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:metadata/SOURCE_MANIFEST.json
all	ntu_viral	official_docs/LICENSE	1079	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:official_docs/LICENSE
all	ntu_viral	official_docs/evaluation_tutorial.md	8070	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:official_docs/evaluation_tutorial.md
all	ntu_viral	official_docs/images/a-Pinhole-camera-model-b-Fisheye-camera-model_W640.jpg	13503	d178cd0d13e94145e2d89af365a40b650bf65a428acc6fac0df06e8c9d2b9fd0	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:official_docs/images/a-Pinhole-camera-model-b-Fisheye-camera-model_W640.jpg
all	ntu_viral	official_docs/images/demo_eee_01.gif	51424950	d6fd7f41aab985ac26d3afda363e7d62da089fb6cd631559e8da3fffc76c2508	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:official_docs/images/demo_eee_01.gif
all	ntu_viral	official_docs/images/demo_nya_01.gif	50627647	4b5371669ce3dae10cc2c43a37c73ce4cb33ea61ee2eb8de6af6f861c4eab304	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:official_docs/images/demo_nya_01.gif
all	ntu_viral	official_docs/images/demo_sbs_01.gif	38102810	9e780bf7db6da625a6525c7a7f1236e85512ea827d73d29dbb1f1cdd1d44053d	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:official_docs/images/demo_sbs_01.gif
all	ntu_viral	official_docs/images/hardware.jpg	184962	e740a79b9734bb23f2dbe47a11ccc641b5f908f5ec465df659fc7174fed799ca	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:official_docs/images/hardware.jpg
all	ntu_viral	official_docs/images/matlabcalibration1.PNG	101399	d23b0af02bd9b9587b4e5abad20fc2b547f29b79c11e687396f0b73ad5153fe2	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:official_docs/images/matlabcalibration1.PNG
all	ntu_viral	official_docs/images/matlabcalibration2.PNG	70530	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:official_docs/images/matlabcalibration2.PNG
all	ntu_viral	official_docs/images/matlabcalibration3.PNG	581712	3c5168bf6ccfe94ac4cf33871708ef0b7b9843655d865b07f76e1794c3562632	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:official_docs/images/matlabcalibration3.PNG
all	ntu_viral	official_docs/images/matlabcalibration4.PNG	293991	795d622824efedea55468df5d746fd1f6403f863c4c508aebc805315116afe7b	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:official_docs/images/matlabcalibration4.PNG
all	ntu_viral	official_docs/images/matlabcalibration5.PNG	622988	f0a728b5efa52c72a56824470c3781d66187aa0561d533a5354976135dea0524	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:official_docs/images/matlabcalibration5.PNG
all	ntu_viral	official_docs/images/ranging_scheme.jpg	47105	46aa8d092cc99171ee4ff7ab97ea1163274e0878190907a2e8b5d02dbb2d8648	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:official_docs/images/ranging_scheme.jpg
all	ntu_viral	official_docs/images/uwb_range_msg.jpg	1827856	c3e88d386461f4849c374faf70494edf1134dabdb3588c41a8d03e7d58009bcc	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:official_docs/images/uwb_range_msg.jpg
all	ntu_viral	official_docs/images/viral_eval_files.png	122044	6e922f692d810285e3370704582eb5cd5a2d274adb8a2ca421684590e24deaa1	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:official_docs/images/viral_eval_files.png
all	ntu_viral	official_docs/images/vn100.jpg	316545	399678b7a9d40a2fb0eb695198e085d7ff2aaaaf8b58bb482f9a68291e5c67bb	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:official_docs/images/vn100.jpg
all	ntu_viral	official_docs/ntuviral_evaluate.ipynb	24141	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:official_docs/ntuviral_evaluate.ipynb
all	ntu_viral	official_docs/requirements.txt	24	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:official_docs/requirements.txt
all	ntu_viral	official_docs/sensor_calibration.md	10124	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:official_docs/sensor_calibration.md
all	ntu_viral	official_docs/sensors_and_usage.md	10731	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:official_docs/sensors_and_usage.md
all	ntu_viral	official_docs/utils/restamp.py	5207	-	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:official_docs/utils/restamp.py
all	ntu_viral	sequences/eee_01.zip	4533480279	7cd8e99bd0ed250638dd98a7157d0ebe334bacd4d3d544f2385540dfb1ad4279	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:sequences/eee_01.zip
all	ntu_viral	sequences/eee_02.zip	3737067770	198a5f404519d11a663865ba4d4ba07a817e44048b13354f3a0c41c29a8083cb	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:sequences/eee_02.zip
all	ntu_viral	sequences/eee_03.zip	1995367404	c842b269cbc9c31788b22095d49e49814499769ef22df9a0728c9e7f2a0f8b94	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:sequences/eee_03.zip
all	ntu_viral	sequences/nya_01.zip	4286607203	3eb0ede8f85148a41799fa10a29325975e3abf5ea58ec8d0f93df26baaa73646	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:sequences/nya_01.zip
all	ntu_viral	sequences/nya_02.zip	4681541004	bda134ecbcbd89eff813742b86fdbd181bf362ccc3dbe9589e7cdc3180a29bfa	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:sequences/nya_02.zip
all	ntu_viral	sequences/nya_03.zip	4406003298	983df97732591eb45004c980d43ab41ee7ac8d5643d41a76d6dc63158be84a9c	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:sequences/nya_03.zip
all	ntu_viral	sequences/rtp_01.zip	5317633190	3762406f683acbaa61f0862790167f1fbf385ea76481c1cc30734aabfc068b60	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:sequences/rtp_01.zip
all	ntu_viral	sequences/rtp_02.zip	5600590974	a824d68a6313eaea55c22c34e80f432ab4517cf66390d2490ee32546e22c26f6	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:sequences/rtp_02.zip
all	ntu_viral	sequences/rtp_03.zip	4266920279	afb7cbbe2b7102278bcb0e2f3aa8582b360a6b434c2e02276d3270e150ac8dec	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:sequences/rtp_03.zip
all	ntu_viral	sequences/sbs_01.zip	3775725925	16063577be7b0d8b6ca89e40cf3ec45d603469048e7d989d25bab88f3950c390	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:sequences/sbs_01.zip
all	ntu_viral	sequences/sbs_02.zip	3763188626	8bb4988e8444a2ec102161f00a9f5c9c7d52fce1341172bb6d97b2267746ee22	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:sequences/sbs_02.zip
all	ntu_viral	sequences/sbs_03.zip	4002686287	b7e2c4a640a16c6c4d40429441124ae670808b511d2024dd1526ef64dd7a0539	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:sequences/sbs_03.zip
all	ntu_viral	sequences/spms_01.zip	5856439436	3a49cbe1bffdb0a18d0b16459545c5b4b24a46674a9da2c071fa14a5044779ca	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:sequences/spms_01.zip
all	ntu_viral	sequences/spms_02.zip	4312000892	3b5fe5a6b9337fca227d62c6c140933c672b58a7d4b03a93cb833f41490aa0af	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:sequences/spms_02.zip
all	ntu_viral	sequences/spms_03.zip	5385740441	45623aa13e7589f3fe6a9c425c43080de87857841b01f1660d0ff624b3e712b5	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:sequences/spms_03.zip
all	ntu_viral	sequences/tnp_01.zip	8736253605	27e3f37f600ddfbf7efc67573f20cf82cb8c22d8e70f87a0580f4c1dee4fd653	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:sequences/tnp_01.zip
all	ntu_viral	sequences/tnp_02.zip	6661876134	b2cae9ea17ba1234e4972b8d539e40f9d35b6e9b5c848249a52c9cb8bda8cf15	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:sequences/tnp_02.zip
all	ntu_viral	sequences/tnp_03.zip	5884321753	0e568377b944d2afed86d79cacfa09bd644e8456b85424484b6aee753e28ac38	hf:Yuan-Shenghai/NTU-VIRAL-DATASET@4a1b869551e7baff1eb6157a61c2f5c3e3d5ffbd:sequences/tnp_03.zip
all	matrixcity_block3	small_city/aerial/test/block_3_test.tar	95078912	0db8ad4d7e37f4599eb9ceae10abe724f8aef5cf9d7edc0dbc175bc89cdc5735	hf:BoDai/MatrixCity@22237509a7a16d5c0136b58b39597629a63b338d:small_city/aerial/test/block_3_test.tar
all	matrixcity_block3	small_city/aerial/train/block_3.tar	855872512	809fe5a6113ceb76abe4e1860c4bd9d72e6064c41af8805766a5112e1e352661	hf:BoDai/MatrixCity@22237509a7a16d5c0136b58b39597629a63b338d:small_city/aerial/train/block_3.tar
all	matrixcity_block3	small_city_depth_float32/aerial/test/block_3_test_depth.tar	107120640	d50dfc2bfcf9437d8b988d98054cfc3a2167112d517ccb9154d23f9f372dda26	hf:BoDai/MatrixCity@22237509a7a16d5c0136b58b39597629a63b338d:small_city_depth_float32/aerial/test/block_3_test_depth.tar
all	matrixcity_block3	small_city_depth_float32/aerial/train/block_3_depth.tar	762603520	2d27221efb7c5d5cda60b573aa66d06a4102caf5c53bcce6bac571ae27fbeadb	hf:BoDai/MatrixCity@22237509a7a16d5c0136b58b39597629a63b338d:small_city_depth_float32/aerial/train/block_3_depth.tar
all	matrixcity_block3	small_city/aerial/pose/block_all/transforms_test.json	429438	-	hf:BoDai/MatrixCity@22237509a7a16d5c0136b58b39597629a63b338d:small_city/aerial/pose/block_all/transforms_test.json
all	matrixcity_block3	small_city/aerial/pose/block_all/transforms_train.json	3305319	-	hf:BoDai/MatrixCity@22237509a7a16d5c0136b58b39597629a63b338d:small_city/aerial/pose/block_all/transforms_train.json
