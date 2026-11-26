# Architecture diagrams

Eraser.io diagram-as-code, one file per version. In eraser.io: new diagram → Flow chart → open the code panel → paste.

**Mermaid:** `scripts/eraser_to_mermaid.py` converts each `.eraser` file to a `.mmd` Mermaid flowchart, refreshes the diagrams below (GitHub renders them), and embeds both into the viewer. Run it after editing a diagram.

**Interactive viewer:** `viewer.html` renders the Mermaid diagrams with Mermaid 11 + the ELK layout (as in codesm's docs): version tabs, pan/zoom, ELK/Dagre and direction toggles, click a box to trace its inputs and outputs, components new since the previous version highlighted, live-editable Mermaid code. Open it in a browser.

| Source | Version |
|---|---|
| `v1-basic-rag.eraser` | V1 basic RAG |
| `v2-grounding-guardrails.eraser` | V2 relevance gate and checks |
| `v3-evidence-first.eraser` | V3 evidence by sentence ID, answer from quotes, removal |
| `v4-hybrid-retrieval.eraser` | V4 vector + keyword, fusion, reranker, neighbours |
| `v5-team-deployment.eraser` | V5 proposed shared deployment |

What changed between versions and why: [../versions.md](../versions.md).

## Diagrams

<!-- mermaid:begin -->
### V1 - Basic local RAG

```mermaid
---
title: V1 - Basic local RAG
---
flowchart LR
  subgraph g0["Data sources"]
    n1["Wikipedia articles"]
    n2["arXiv PDFs"]
    n3["NASA images and captions"]
  end
  subgraph g4["Ingestion"]
    n5["Loaders"]
    n6["Chunker 1200 chars"]
    n7["Embedder nomic-embed-text"]
  end
  n8[("Postgres pgvector")]
  subgraph g9["Query path"]
    n10["User"]
    n11["Chat CLI"]
    n12["Query embedder"]
    n13["Top-6 cosine search"]
    n14["Image boost"]
    n15["LLM qwen3-4b-instruct"]
  end
  n1 --> n5
  n2 --> n5
  n3 --> n5
  n5 --> n6
  n6 --> n7
  n7 -->|"vectors + text + metadata"| n8
  n10 -->|"question"| n11
  n11 --> n12
  n12 --> n13
  n13 <--> n8
  n13 -->|"picture questions"| n14
  n14 -->|"sources + prompt"| n15
  n15 -->|"answer + citations"| n10
  classDef gray fill:#6b728022,stroke:#6b7280,stroke-width:1.5px
  classDef green fill:#2b835222,stroke:#2b8352,stroke-width:1.5px
  classDef blue fill:#2c66c422,stroke:#2c66c4,stroke-width:1.5px
  classDef purple fill:#7148c022,stroke:#7148c0,stroke-width:1.5px
  classDef orange fill:#bf671622,stroke:#bf6716,stroke-width:1.5px
  classDef red fill:#c2343422,stroke:#c23434,stroke-width:1.5px
  class n1,n2,n3 gray
  class n5,n6,n7 green
  class n8 blue
  class n10,n11,n12,n13,n14,n15 purple
  style g0 fill:#6b72800f,stroke:#6b7280,stroke-dasharray:5 4
  style g4 fill:#2b83520f,stroke:#2b8352,stroke-dasharray:5 4
  style g9 fill:#7148c00f,stroke:#7148c0,stroke-dasharray:5 4
```

### V2 - Grounding guardrails

```mermaid
---
title: V2 - Grounding guardrails
---
flowchart LR
  n0[("Postgres pgvector")]
  subgraph g1["Query path"]
    n2["User"]
    n3["Query embedder"]
    n4["Top-6 cosine search"]
    n5{"Relevance gate"}
    n6["Refuse without LLM"]
    n7["Strict prompt temp 0"]
    n8["LLM qwen3-4b-instruct"]
  end
  subgraph g9["Checks"]
    n10["Citation check"]
    n11["Optional LLM fact check"]
    n12["Warnings shown to user"]
  end
  subgraph g13["Evaluation"]
    n14["Eval set facts images traps"]
  end
  n2 -->|"question"| n3
  n3 --> n4
  n4 <--> n0
  n4 --> n5
  n5 -->|"best score below 0.68"| n6
  n5 -->|"relevant"| n7
  n7 --> n8
  n8 -->|"answer"| n10
  n10 -->|"verify flag"| n11
  n11 --> n12
  n10 -->|"answer + sources"| n2
  n14 -->|"calibrates threshold"| n5
  classDef gray fill:#6b728022,stroke:#6b7280,stroke-width:1.5px
  classDef green fill:#2b835222,stroke:#2b8352,stroke-width:1.5px
  classDef blue fill:#2c66c422,stroke:#2c66c4,stroke-width:1.5px
  classDef purple fill:#7148c022,stroke:#7148c0,stroke-width:1.5px
  classDef orange fill:#bf671622,stroke:#bf6716,stroke-width:1.5px
  classDef red fill:#c2343422,stroke:#c23434,stroke-width:1.5px
  class n0 blue
  class n2,n3,n4,n7,n8 purple
  class n5,n10,n11,n12 orange
  class n6 red
  class n14 gray
  style g1 fill:#7148c00f,stroke:#7148c0,stroke-dasharray:5 4
  style g9 fill:#bf67160f,stroke:#bf6716,stroke-dasharray:5 4
  style g13 fill:#6b72800f,stroke:#6b7280,stroke-dasharray:5 4
```

### V3 - Evidence-first answering

```mermaid
---
title: V3 - Evidence-first answering
---
flowchart LR
  n0[("Postgres pgvector")]
  subgraph g1["Understand"]
    n2["User"]
    n3["Chat history"]
    n4["Follow-up rewriter"]
  end
  subgraph g5["Retrieve"]
    n6["Top-6 cosine search"]
    n7{"Relevance gate"}
  end
  subgraph g8["Evidence"]
    n9["Sentence numbering"]
    n10["LLM selects sentence IDs"]
    n11["Code looks up exact sentences"]
  end
  subgraph g12["Answer"]
    n13["LLM answers from quotes only"]
  end
  subgraph g14["Verify"]
    n15["Citation check"]
    n16["Batch LLM fact check"]
    n17["Remove unsupported sentences"]
  end
  n18["Refusal"]
  n2 -->|"question"| n4
  n3 --> n4
  n4 -->|"standalone question"| n6
  n6 <--> n0
  n6 --> n7
  n7 -->|"nothing relevant"| n18
  n7 --> n9
  n9 --> n10
  n10 -->|"IDs like 2.5"| n11
  n11 -->|"no valid evidence"| n18
  n11 -->|"verified quotes"| n13
  n13 --> n15
  n15 --> n16
  n16 --> n17
  n17 -->|"nothing supported left"| n18
  n17 -->|"answer + evidence + sources"| n2
  classDef gray fill:#6b728022,stroke:#6b7280,stroke-width:1.5px
  classDef green fill:#2b835222,stroke:#2b8352,stroke-width:1.5px
  classDef blue fill:#2c66c422,stroke:#2c66c4,stroke-width:1.5px
  classDef purple fill:#7148c022,stroke:#7148c0,stroke-width:1.5px
  classDef orange fill:#bf671622,stroke:#bf6716,stroke-width:1.5px
  classDef red fill:#c2343422,stroke:#c23434,stroke-width:1.5px
  class n0,n6 blue
  class n2,n3,n4,n13 purple
  class n7,n15,n16,n17 orange
  class n9,n10,n11 green
  class n18 red
  style g1 fill:#7148c00f,stroke:#7148c0,stroke-dasharray:5 4
  style g5 fill:#2c66c40f,stroke:#2c66c4,stroke-dasharray:5 4
  style g8 fill:#2b83520f,stroke:#2b8352,stroke-dasharray:5 4
  style g12 fill:#7148c00f,stroke:#7148c0,stroke-dasharray:5 4
  style g14 fill:#bf67160f,stroke:#bf6716,stroke-dasharray:5 4
```

### V4 - Hybrid retrieval with reranking

```mermaid
---
title: V4 - Hybrid retrieval with reranking
---
flowchart LR
  subgraph g0["Postgres"]
    n1[("HNSW vector index")]
    n2[("Full-text GIN index")]
    n3[("Chunk table")]
  end
  subgraph g4["Retrieve"]
    n5["Standalone question"]
    n6["Vector search top 30"]
    n7["Keyword search top 30"]
    n8["RRF fusion"]
    n9["Reranker jina-turbo"]
    n10{"Reranker gate"}
    n11["Add neighbouring chunks"]
  end
  subgraph g12["Downstream"]
    n13["Evidence selection and verified answer"]
  end
  n14["Refusal"]
  n5 --> n6
  n5 --> n7
  n6 <--> n1
  n7 <--> n2
  n6 --> n8
  n7 --> n8
  n8 -->|"24 candidates"| n9
  n9 -->|"top 8"| n10
  n10 -->|"best score below 0.3"| n14
  n10 --> n11
  n11 <--> n3
  n11 -->|"same as V3"| n13
  classDef gray fill:#6b728022,stroke:#6b7280,stroke-width:1.5px
  classDef green fill:#2b835222,stroke:#2b8352,stroke-width:1.5px
  classDef blue fill:#2c66c422,stroke:#2c66c4,stroke-width:1.5px
  classDef purple fill:#7148c022,stroke:#7148c0,stroke-width:1.5px
  classDef orange fill:#bf671622,stroke:#bf6716,stroke-width:1.5px
  classDef red fill:#c2343422,stroke:#c23434,stroke-width:1.5px
  class n1,n2,n3,n5,n6,n7,n8,n9,n11 blue
  class n10 orange
  class n13 green
  class n14 red
  style g0 fill:#2c66c40f,stroke:#2c66c4,stroke-dasharray:5 4
  style g4 fill:#2c66c40f,stroke:#2c66c4,stroke-dasharray:5 4
  style g12 fill:#2b83520f,stroke:#2b8352,stroke-dasharray:5 4
```

### V5 - Shared team deployment (proposed)

```mermaid
---
title: V5 - Shared team deployment (proposed)
---
flowchart LR
  subgraph g0["Teams"]
    n1["Team A apps"]
    n2["Team B chatbot"]
  end
  subgraph g3["Ingestion workers"]
    n4["Document watcher"]
    n5["Ingest pipeline"]
  end
  subgraph g6["Shared services"]
    n7["RAG API service"]
    n8["Embedding server"]
    n9["LLM server"]
    n10["Reranker"]
  end
  subgraph g11["Storage"]
    n12[("Object storage MinIO")]
    n13[("Postgres pgvector replicas")]
  end
  n4 -->|"new or changed files"| n5
  n5 -->|"raw PDFs and images"| n12
  n5 --> n8
  n5 -->|"chunks + vectors + team ACL"| n13
  n1 -->|"question + team"| n7
  n2 --> n7
  n7 --> n8
  n7 <-->|"filtered by team and embed model"| n13
  n7 --> n10
  n7 --> n9
  n7 -->|"grounded answer"| n1
  classDef gray fill:#6b728022,stroke:#6b7280,stroke-width:1.5px
  classDef green fill:#2b835222,stroke:#2b8352,stroke-width:1.5px
  classDef blue fill:#2c66c422,stroke:#2c66c4,stroke-width:1.5px
  classDef purple fill:#7148c022,stroke:#7148c0,stroke-width:1.5px
  classDef orange fill:#bf671622,stroke:#bf6716,stroke-width:1.5px
  classDef red fill:#c2343422,stroke:#c23434,stroke-width:1.5px
  class n1,n2 gray
  class n4,n5 green
  class n7,n8,n9,n10 blue
  class n12,n13 purple
  style g0 fill:#6b72800f,stroke:#6b7280,stroke-dasharray:5 4
  style g3 fill:#2b83520f,stroke:#2b8352,stroke-dasharray:5 4
  style g6 fill:#2c66c40f,stroke:#2c66c4,stroke-dasharray:5 4
  style g11 fill:#7148c00f,stroke:#7148c0,stroke-dasharray:5 4
```
<!-- mermaid:end -->
