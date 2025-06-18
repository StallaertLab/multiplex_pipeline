# 🎯 Channel Selection Logic

The pipeline supports fine-grained control over which imaging channels are included in processing. This is essential because:

- The same marker may be imaged multiple times across rounds (e.g., re-staining or optimization).
- DAPI is typically acquired in every round for registration but usually only one version is needed.

The selection process follows this logic:

### 1. **Default Behavior (if no overrides)**
- For each marker imaged in multiple rounds, the **latest round** is used by default.
- For DAPI, only `001_DAPI` is included unless specified otherwise.

### 2. **Using `include_channels`**
- This is a list of fully qualified channel names like `002_CD44`, `001_DAPI`.
- If set, only these channels are included for a given marker — **they override automatic selection**.
- Use this to **force inclusion of earlier rounds** or **include duplicates** for comparison.

### 3. **Using `exclude_channels`**
- This is a list of full channel names to skip.
- If `include_channels` is not set, `exclude_channels` can be used to **remove undesired versions**.
- Example: to exclude `003_pRB` in favor of earlier versions (or none).

### 4. **Using `use_channels`**
- This is a list of **base marker names** (like `DAPI`, `pRB`, `CD44`) **after stripping the round prefix**.
- After all other filtering, `use_channels` is applied as a final filter.
- Use it to **narrow the final channel set to specific markers**, regardless of which round was selected.

---

## Examples

### Example 1: Default automatic selection
```yaml
include_channels: []
exclude_channels: []
use_channels: []
```

* Keeps only the **latest round per marker**, and `001_DAPI`.

---

### Example 2: Force earlier pRB round to be used

```yaml
include_channels: ["001_pRB"]
use_channels: []
```

* `001_pRB` is used even if `003_pRB` exists.

---

### Example 3: Exclude a problematic round

```yaml
exclude_channels: ["003_CD44"]
```

* Automatically selects an earlier round (if available) for CD44.

---

### Example 4: Only process DAPI and CD44

```yaml
use_channels: ["DAPI", "CD44"]
```

* Filters final output to only include these two base markers.

---

### Conflicts and Priority

* If a channel is listed in both `include_channels` and `exclude_channels`, a `ValueError` is raised.
* `use_channels` is applied **last**, on the base names after channel selection.