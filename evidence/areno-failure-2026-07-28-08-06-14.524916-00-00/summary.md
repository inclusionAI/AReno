# AReno Failure Bundle

**Timestamp**: 2026-07-28T08:06:14.524916+00:00
**AReno Version**: 0.0.6
**Python**: 3.14.6 (v3.14.6:c63aec69bd5, Jun 10 2026, 08:07:54) [Clang 21.0.0 (clang-2100.1.1.101)]
**Platform**: macOS-26.5.2-arm64-arm-64bit-Mach-O

## Command
```
--traceback-file crash.log --output-dir ./evidence/
```

## Error
**Type**: `RuntimeError`
**Message**: Traceback (most recent call last):
  File "train.py", line 42, in <module>
ValueError: CUDA out of memory


### Traceback (most recent call last)
```
RuntimeError: Traceback (most recent call last):
  File "train.py", line 42, in <module>
ValueError: CUDA out of memory
```

## GPU
```json
{
  "available": false
}
```

## CUDA
```json
{
  "cuda_home": null,
  "nvcc_path": null
}
```

## Process
```json
{
  "pid": 40357,
  "ppid": 39301,
  "cwd": "/Users/naxida/Desktop/AReno",
  "executable": "/Users/naxida/Desktop/AReno/.venv/bin/python"
}
```
