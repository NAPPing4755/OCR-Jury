# Error analysis - dev split, jury `tesseract+easyocr+doctr+trocr`

671 lines; 444 (66.2%) have at least one error; 1388 character edits over 36522 reference characters (CER 3.80%).

| Error type | Lines | Share of error characters | Example (ground truth -> jury) |
|---|---|---|---|
| multi-character misread | 111 | 42.4% | `successfully` -> `fuccefsfull.,` |
| digits differ (misread numbers, or letters read as digits) | 97 | 31.2% | `Some skill in these abstract...` -> `Some 1kill in these abstract...` |
| 1-2 character misread | 103 | 12.1% | `...n also and Constantine, two ...` -> `...n also and Conftaatine, two ...` |
| punctuation / spacing only | 79 | 8.9% | `culcated a maxim, which, tho...` -> `culcated.a maxim, which, tho...` |
| long-s / f confusion left unrepaired | 40 | 3.8% | `...t be won, and so much author...` -> `...t be won, and fo much author...` |
| very short line (<=3 chars) | 9 | 1.1% | `1.` -> `1..` |
| letter case only | 5 | 0.4% | `...ung king, ib. joins the Engl...` -> `...ung king, ib. Joins the Engl...` |
