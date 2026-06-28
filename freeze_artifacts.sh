#!/bin/bash
echo "--- Artifact Freeze & Hashing ---"
MANIFEST="artifact_manifest.json"
echo "{" > $MANIFEST
echo '  "models": {' >> $MANIFEST
first=1
for f in models/*; do
  if [ -f "$f" ]; then
    hash=$(sha256sum "$f" | awk '{print $1}')
    if [ $first -eq 1 ]; then
      echo '    "'$f'": "'$hash'"' >> $MANIFEST
      first=0
    else
      echo '    ,"'$f'": "'$hash'"' >> $MANIFEST
    fi
  fi
done
echo '  },' >> $MANIFEST

echo '  "reports": {' >> $MANIFEST
first=1
for f in reports/*; do
  if [ -f "$f" ]; then
    hash=$(sha256sum "$f" | awk '{print $1}')
    if [ $first -eq 1 ]; then
      echo '    "'$f'": "'$hash'"' >> $MANIFEST
      first=0
    else
      echo '    ,"'$f'": "'$hash'"' >> $MANIFEST
    fi
  fi
done
echo '  }' >> $MANIFEST
echo "}" >> $MANIFEST

echo "Manifest generated at $MANIFEST"
