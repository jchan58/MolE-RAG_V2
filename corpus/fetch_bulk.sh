#!/usr/bin/env bash
set -u
BASE="https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras"
DST="/scratch/bhrm/jchan5/exp70_bbbp_pmc_collection/bulk_ftp"
cd "$DST" || exit 1
for f in CID-InChI-Key.gz CID-Title.gz CID-IUPAC.gz CID-Synonym-filtered.gz; do
  for attempt in 1 2 3 4 5; do
    echo "[$(date -u +%H:%M:%S)] downloading $f (attempt $attempt)"
    wget -c -q --tries=3 --timeout=120 "$BASE/$f" -O "$DST/$f" && \
    wget -c -q --tries=3 --timeout=60 "$BASE/$f.md5" -O "$DST/$f.md5" && break
    echo "[$(date -u +%H:%M:%S)] retN $f failed, sleeping"; sleep 10
  done
  echo "[$(date -u +%H:%M:%S)] done $f size=$(stat -c%s "$DST/$f" 2>/dev/null)"
done
# verify md5s
echo "[$(date -u +%H:%M:%S)] verifying md5"
cd "$DST"
for f in CID-InChI-Key.gz CID-Title.gz CID-IUPAC.gz CID-Synonym-filtered.gz; do
  if [ -f "$f.md5" ]; then md5sum -c "$f.md5" 2>&1 | sed "s/^/md5: /"; fi
done
echo "[$(date -u +%H:%M:%S)] ALL_BULK_DONE"
