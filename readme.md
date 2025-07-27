# 🤖 Bid-Analysis-Agent: An End-to-End AI System

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![F1 Score](https://img.shields.io/badge/Agent_F1_Score-0.964-blue)
![Tech Stack](https://img.shields.io/badge/Tech-PyTorch_|_Transformers_|_LangGraph-orange)
![Deployment](https://img.shields.io/badge/Deployment-FastAPI_|_ONNX-purple)

An End-to-End AI agent system for Korean public procurement analysis, achieving a 96.4% F1-score by leveraging a fine-tuned RoBERTa, RAG, and LLMs. This project demonstrates a full-cycle development process from ideation and data collection to model optimization and API deployment.

<!-- A GIF showcasing the Agent System or the final API in action would go here. -->

## 1. Overview: From a Business Problem to a Production API

This project was born from a real-world business challenge: the inefficient and error-prone process of manually reviewing hundreds of public procurement bids daily at a civil engineering firm. This manual process not only consumed significant human resources but also risked missing critical business opportunities.

This repository documents the journey of building an End-to-End solution single-handedly. It covers everything from **data collection and refinement** and **core AI model development and optimization**, to the final deployment of a **real-time, production-ready API**.

The project culminates in two key deliverables:
1.  **A Stable Production API:** A highly optimized dual-model system built for speed, stability, and immediate business integration.
2.  **A High-Performance R&D Agent:** An intelligent agent system designed to push the boundaries of accuracy, mimicking the decision-making process of a human expert.

---

## 2. Solution Architecture

The entire project is structured as a systematic pipeline, flowing from data foundation to model development, optimization, and final service delivery.

```mermaid
graph TD
    subgraph "Phase 1: Data Foundation"
        A[<b>RAW DATA</b><br/>Procurement Bid Collection] -->|'Cleanse & Standardize'| B(<b>High-Quality Labeled Dataset</b>);
        B -->|'Stratified Sampling'| C[Train/Valid/Test Sets];
    end

    subgraph "Phase 2: Core AI Model Development"
        C --> D[<b>Train Primary Review AI</b><br/>Binary Classifier (FT RoBERTa+LoRA)];
        C --> E[<b>Train Secondary Review AI</b><br/>Multi-Class Classifier (FT RoBERTa+LoRA)];
        C --> F[<b>Build Knowledge Base</b><br/>Semantic Search DB (SBERT + Faiss)];
    end

    subgraph "Phase 3: Production Optimization"
       D --> G[<b>Model Quantization & Optimization</b><br/>ONNX Conversion + INT8 Quantization];
       E --> G;
       G --> H[<b>Optimized Models</b><br/>2.35x Faster, 75% Smaller];
    end
    
    H --> I{<b>Strategic Fork</b>};

    subgraph "<b>Path 1: Production API System</b>"
        I --"> <b>Focus: Speed & Stability</b>" --> J[<b>Build Real-time Service</b><br/>Dual-Model API with FastAPI];
        J --> K[<b>FINAL API SERVICE</b><br/>POST /classify_batch<br/><i>Delivers reliable business value</i>];
    end

    subgraph "<b>Path 2: R&D Agent System</b>"
        I --"> <b>Focus: Peak Accuracy</b>" --> L[<b>Build Expert Committee AI</b><br/>3-Agent System (FT Model + RAG + LLM)];
        F --> L;
        D --> L;
        L --> M[<b>Ablation Study</b><br/>FT Model vs. Agent System];
        M --> N[<b>Achieved 0.964 F1-Score</b><br/><i>Proved 88% error reduction</i>];
    end

    %% Styling for better readability
    style A fill:#0277BD,color:#fff,stroke:#F5F5F5,stroke-width:2px
    style B fill:#039BE5,color:#fff,stroke:#F5F5F5,stroke-width:2px
    style C fill:#29B6F6,color:#000,stroke:#333,stroke-width:2px
    style D fill:#388E3C,color:#fff,stroke:#F5F5F5,stroke-width:2px
    style E fill:#4CAF50,color:#fff,stroke:#F5F5F5,stroke-width:2px
    style F fill:#66BB6A,color:#000,stroke:#333,stroke-width:2px
    style G fill:#F9A825,color:#000,stroke:#333,stroke-width:2px
    style H fill:#FBC02D,color:#000,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
    style I fill:#8E24AA,color:#fff,stroke:#F5F5F5,stroke-width:4px
    style J fill:#D32F2F,color:#fff,stroke:#F5F5F5,stroke-width:2px
    style K fill:#F44336,color:#fff,stroke:#F5F5F5,stroke-width:4px
    style L fill:#303F9F,color:#fff,stroke:#F5F5F5,stroke-width:2px
    style M fill:#3F51B5,color:#fff,stroke:#F5F5F5,stroke-width:2px
    N fill:#5C6BC0,color:#fff,stroke:#F5F5F5,stroke-width:3px,stroke-dasharray: 5 5
```

---

## 3. Performance Analysis: From Simple Automation to an Intelligent Expert System

This project evolved in two stages: first, building a robust automation system, and second, proving the potential of a more intelligent expert system that overcomes the limitations of the first.

#### **Evolution 1: The Robust Automator**
*   **Reliable Task Handling:** A core engine was developed by fine-tuning `klue/roberta-large`, achieving a high F1-Score of **0.97** in determining bid eligibility. This model forms the backbone of the fast and reliable production API.

#### **Evolution 2: The Intelligent Expert**
*   **Mimicking Human Decision-Making:** Simple fine-tuned models can be brittle on ambiguous, boundary-case problems. To solve this, a 3-agent system was designed, where a **Fine-Tuned Model (Initial Review) + RAG (Case Search) + LLM (Final Judgment)** collaborate to make a decision.
*   **Breakthrough Performance:** This agent-based approach dramatically improved judgment accuracy on complex cases, boosting the overall system's **F1-Score from a baseline of 0.7045 to 0.9639**. This represents an **88% reduction in the error rate** compared to the standalone fine-tuned model on challenging data.

*(Insert your F1-Score comparison graph here. Ensure labels are in English.)*
![F1-Score Comparison](path/to/your/f1_score_comparison_graph.png)

---

## 4. Optimization for Production

A powerful model is useless if it's not fast and efficient in a real-world service environment. Targeting a CPU-only server deployment, the trained PyTorch models were optimized through **ONNX conversion and INT8 quantization**.

*(Insert your quantization performance comparison table here.)*
| Metric          | FP32 PyTorch (Baseline) | INT8 ONNX (Quantized) | Delta (Change)                     |
|:----------------|------------------------:|----------------------:|:-----------------------------------|
| **F1-Score**    | **0.9719**              | **0.9739**           | **+0.0020 (+0.20%)**               |
| Model Size (MB) | 1280.75                 | 323.24               | -74.76%                            |
| Latency (ms)    | 103.98                  | 44.15                | **-57.54% (2.35x faster)**         |

**Analysis:** Remarkably, the optimization process resulted in **zero performance degradation**—in fact, the F1-Score slightly increased. We achieved a **2.35x speedup in inference and a 75% reduction in model size**, making the system perfectly viable for a cost-effective CPU-only API service.

---

## 5. Project Structure & How to Run

The project is designed with a modular, role-based structure for clarity and scalability.

### Project Structure
```
Bid-Analysis-Agent/
├── src/
│   ├── core_training/          # The "Brain Factory": Scripts for training core models
│   ├── production_api/       # The "Workplace": Production-ready API logic
│   ├── research_agent_system/  # The "Lab": R&D agent system logic
│   └── shared_utils/           # The "Heart": Common utilities (data, models, etc.)
├── data/
├── notebooks/                  # EDA and initial experiments
├── output/                     # All artifacts: models, results, vector DB
├── config.py
└── README.md
```

### How to Reproduce
1.  **Setup Environment:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Process Data:**
    ```bash
    python src/shared_utils/data_processing.py
    ```
3.  **Train Core Models:**
    ```bash
    python src/core_training/train_binary.py
    python src/core_training/train_multiclass.py
    ```
4.  **Run API Server:**
    ```bash
    python src/production_api/main.py
    ```

---

## 6. Tech Stack
*   **Language:** Python
*   **Core Libraries:** PyTorch, Transformers, PEFT(LoRA), LangGraph
*   **API & Deployment:** FastAPI, Uvicorn, ONNX Runtime
*   **Data Handling:** Pandas, Faiss, Scikit-learn
*   **Environment:** Conda

---

## License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

## Author
*   **Harim Choi (HarimxChoi)**
*   **Email:** 2.harim.choi@gmail.com
*   **LinkedIn:** 
*   **Medium:** 