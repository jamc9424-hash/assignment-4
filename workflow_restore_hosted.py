# Restore the already completed hosted runs; verify immutable repository copies before reuse.
from urllib.request import urlopen
HOSTED_COMMIT = "9bc981c98ed51f32de5cafca86a960754aa6c416"
HOSTED_HASHES = {"evaluation_package_versions.json": "bd16e14b0cfb4708602f1cd7681647d2b5c48d0115b4107ff11f716e8468d861", "few_shot_predictions.jsonl": "4b2dd1d2a588b2abd898de3ce5401a5bccc89e253e89081aeb3bf2c3d0ffd94c", "few_shot_responses.jsonl": "a5aa9e2445ab1474b2a0a646e4001ad09301102cbe27040dd08480e02826ea93", "few_shot_run.json": "ff65f4f3d4bc10c2acdbec23039bc59f8c873a9ec97b929f656e54ed75c4a4bb", "hosted_evaluation.json": "fcee675262c0383ef0d6289bea142f0b2f0ed05ad1f519ab611954e500420bc0", "hosted_runs.json": "93d44fcc996ae13a50f1609e9c1ffd1cbdc833b7e3f9509003b7179e8dad9193", "request_manifest.json": "6d138a24225cda9a0cdcde5e7cdec2083cb7a7331045c61582a3e8123c060bc4", "split_audit.json": "dd8417276a5495832d2c1ddda12a9ba12c2adbafdee881d67d8457502a4694c6", "zero_shot_predictions.jsonl": "4d20b98d4733b0cede9c8ba26c172071cc05cc7bb964550726087582c71b19b4", "zero_shot_responses.jsonl": "34e27acb2a44f0d8435b18a26360b28e78a594b3473492069a10ed08413580ad", "zero_shot_run.json": "6db713760d6fd249015a3b32b954e6632c48eb79968c9e339754e7bef1c818a1"}
local_hosted = WORK / "hosted_local"
local_hosted.mkdir(exist_ok=True)
for name, expected_hash in HOSTED_HASHES.items():
    url = f"https://raw.githubusercontent.com/jamc9424-hash/assignment-4/{HOSTED_COMMIT}/evidence/hosted_local/{name}"
    with urlopen(url, timeout=60) as response:
        data = response.read()
    assert hashlib.sha256(data).hexdigest() == expected_hash, (name, "changed artifact")
    (local_hosted/name).write_bytes(data)
print("PASS: hosted evidence restored from pinned GitHub commit; every file hash verified.")
