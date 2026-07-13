# CyberWatch input data

Place network-flow or malware-feature CSV files in this folder.

Docker mounts this folder read-only at `/data`. Example:

```powershell
docker compose run --rm cyberwatch network --csv /data/flows.csv
```
