# Graph Data — roadNet-CA

## Dataset: California Road Network (roadNet-CA)
**Source:** [Stanford SNAP](https://snap.stanford.edu/data/roadNet-CA.html)  
**File:** `roadNet-CA.txt`  
**Format:** Tab-separated edge list, comment lines start with `#`

---

## Download Instructions

### Option 1 — Direct wget (Linux / macOS / WSL)
```bash
mkdir -p data
wget -P data/ https://snap.stanford.edu/data/roadNet-CA.txt.gz
gunzip data/roadNet-CA.txt.gz
```

### Option 2 — curl (Windows PowerShell)
```powershell
New-Item -ItemType Directory -Force -Path data
Invoke-WebRequest -Uri "https://snap.stanford.edu/data/roadNet-CA.txt.gz" `
    -OutFile "data\roadNet-CA.txt.gz"
# Then use 7-Zip, WinRAR, or the built-in tar to extract:
tar -xzf data\roadNet-CA.txt.gz -C data\
```

### Option 3 — Python one-liner
```python
import urllib.request, gzip, shutil
urllib.request.urlretrieve(
    "https://snap.stanford.edu/data/roadNet-CA.txt.gz",
    "data/roadNet-CA.txt.gz"
)
with gzip.open("data/roadNet-CA.txt.gz", "rb") as f_in, \
     open("data/roadNet-CA.txt", "wb") as f_out:
    shutil.copyfileobj(f_in, f_out)
```

---

## Dataset Statistics
| Property          | Value       |
|-------------------|-------------|
| Vertices          | 1,965,206   |
| Edges (undirected)| 2,766,607   |
| Average degree    | 2.82        |
| Diameter          | ~849        |
| File size (raw)   | ~73 MB      |

---

## Smaller Test Graphs (for quick smoke tests)

You can also use any other SNAP-format edge list.  
Good small alternatives from SNAP:
- **roadNet-PA** (1.09M vertices, 1.54M edges)  
  `https://snap.stanford.edu/data/roadNet-PA.txt.gz`
- **roadNet-TX** (1.38M vertices, 1.92M edges)  
  `https://snap.stanford.edu/data/roadNet-TX.txt.gz`

For a tiny smoke-test, create a synthetic graph:
```julia
# gen_test_graph.jl
open("data/test_graph.txt", "w") do f
    println(f, "# Synthetic chain graph for smoke test")
    for i in 0:999
        println(f, "$i\t$(i+1)")
    end
end
```
Then run:
```bash
mpirun -np 2 julia --project=. --threads 2 src/main.jl --graph data/test_graph.txt
```
