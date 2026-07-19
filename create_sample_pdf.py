"""
create_sample_pdf.py — Generate a multi-page sample PDF for testing chunking.

This creates a realistic ~5 page PDF about Machine Learning basics,
so the chunking output will make sense and be relatable to your ML background.

Run once: python create_sample_pdf.py
Then run: python step2_chunking.py
"""

import sys
from fpdf import FPDF


def create_sample_pdf():
    # Reconfigure stdout to support UTF-8 print (emojis on Windows)
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Page 1: Introduction to Machine Learning ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "Introduction to Machine Learning", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 7, (
        "Machine Learning (ML) is a subset of artificial intelligence that focuses on "
        "building systems that learn from data. Unlike traditional programming where rules "
        "are explicitly coded, ML algorithms discover patterns in data and use them to make "
        "predictions or decisions.\n\n"
        "The field has grown enormously since the 2010s, driven by three key factors: "
        "the availability of large datasets, improvements in computing power (especially GPUs), "
        "and breakthroughs in algorithm design. Today, ML powers everything from recommendation "
        "systems at Netflix to self-driving cars at Waymo.\n\n"
        "There are three main types of machine learning:\n\n"
        "1. Supervised Learning: The algorithm learns from labeled training data. Each training "
        "example consists of an input (features) and a desired output (label). The goal is to "
        "learn a mapping function from inputs to outputs. Common algorithms include Linear "
        "Regression, Decision Trees, Random Forests, and Support Vector Machines (SVMs).\n\n"
        "2. Unsupervised Learning: The algorithm works with unlabeled data and tries to find "
        "hidden patterns or structures. Clustering algorithms like K-Means and DBSCAN group "
        "similar data points together. Dimensionality reduction techniques like PCA (Principal "
        "Component Analysis) reduce the number of features while preserving important information.\n\n"
        "3. Reinforcement Learning: An agent learns to make decisions by interacting with an "
        "environment. The agent receives rewards or penalties based on its actions and learns "
        "a policy that maximizes cumulative reward over time. This approach has achieved remarkable "
        "results in game playing (AlphaGo) and robotics."
    ))

    # ── Page 2: Neural Networks and Deep Learning ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "Neural Networks and Deep Learning", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 7, (
        "Deep Learning is a subset of machine learning based on artificial neural networks. "
        "A neural network consists of layers of interconnected nodes (neurons) that process "
        "information. The basic architecture includes an input layer, one or more hidden layers, "
        "and an output layer.\n\n"
        "Each connection between neurons has a weight, and each neuron has a bias. During forward "
        "propagation, inputs are multiplied by weights, summed with biases, and passed through "
        "activation functions like ReLU (Rectified Linear Unit) or Sigmoid. The network's output "
        "is compared to the desired output using a loss function.\n\n"
        "Backpropagation is the algorithm used to train neural networks. It calculates the gradient "
        "of the loss function with respect to each weight using the chain rule of calculus. These "
        "gradients are then used by optimization algorithms like Stochastic Gradient Descent (SGD) "
        "or Adam to update the weights and minimize the loss.\n\n"
        "Convolutional Neural Networks (CNNs) are specialized for processing grid-like data such as "
        "images. They use convolutional layers that apply filters to detect features like edges, "
        "textures, and shapes. Pooling layers reduce spatial dimensions while retaining important "
        "features. CNNs have revolutionized computer vision tasks including image classification, "
        "object detection, and image segmentation.\n\n"
        "Recurrent Neural Networks (RNNs) are designed for sequential data like text and time series. "
        "They maintain a hidden state that captures information from previous time steps. However, "
        "basic RNNs struggle with long-range dependencies due to the vanishing gradient problem. "
        "Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs) address this by "
        "introducing gating mechanisms that control information flow."
    ))

    # ── Page 3: Natural Language Processing ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "Natural Language Processing", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 7, (
        "Natural Language Processing (NLP) is a field at the intersection of computer science, "
        "artificial intelligence, and linguistics. It focuses on enabling computers to understand, "
        "interpret, and generate human language.\n\n"
        "Traditional NLP relied on hand-crafted rules and statistical methods. Bag of Words (BoW) "
        "and TF-IDF (Term Frequency-Inverse Document Frequency) were common text representation "
        "methods. These approaches treated words independently and lost all information about word "
        "order and context.\n\n"
        "Word embeddings like Word2Vec and GloVe represented words as dense vectors in a continuous "
        "space, capturing semantic relationships. The famous example 'king - man + woman = queen' "
        "demonstrated that these vectors encode meaningful relationships.\n\n"
        "The Transformer architecture, introduced in the 2017 paper 'Attention Is All You Need', "
        "revolutionized NLP. Transformers use self-attention mechanisms that allow each word to "
        "attend to every other word in the sequence, capturing long-range dependencies effectively. "
        "This architecture is the foundation of modern language models.\n\n"
        "Large Language Models (LLMs) like GPT (Generative Pre-trained Transformer) and BERT "
        "(Bidirectional Encoder Representations from Transformers) are trained on massive text "
        "corpora. GPT models are decoder-only transformers trained to predict the next token, while "
        "BERT uses masked language modeling where random tokens are masked and the model predicts them. "
        "These models can be fine-tuned for specific tasks or used with prompting techniques.\n\n"
        "Retrieval-Augmented Generation (RAG) combines the strengths of retrieval systems and "
        "generative models. Instead of relying solely on the LLM's training data, RAG retrieves "
        "relevant documents from an external knowledge base and includes them in the prompt. This "
        "approach reduces hallucinations and allows the model to access up-to-date information."
    ))

    # ── Page 4: Model Evaluation and Metrics ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "Model Evaluation and Metrics", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 7, (
        "Evaluating machine learning models is crucial to understand their performance and "
        "identify areas for improvement. Different tasks require different evaluation metrics.\n\n"
        "For classification tasks, common metrics include:\n\n"
        "Accuracy: The ratio of correctly predicted instances to total instances. While intuitive, "
        "accuracy can be misleading with imbalanced datasets. If 95% of samples are class A, a model "
        "that always predicts A achieves 95% accuracy but is useless.\n\n"
        "Precision: Of all positive predictions, how many were actually positive? Precision = TP/(TP+FP). "
        "High precision means few false positives. Important when false positives are costly, like spam "
        "detection (you don't want to flag legitimate emails).\n\n"
        "Recall (Sensitivity): Of all actual positives, how many were correctly identified? "
        "Recall = TP/(TP+FN). High recall means few false negatives. Critical in medical diagnosis "
        "where missing a disease (false negative) can be life-threatening.\n\n"
        "F1 Score: The harmonic mean of precision and recall. F1 = 2 * (Precision * Recall) / "
        "(Precision + Recall). Useful when you need a balance between precision and recall.\n\n"
        "For regression tasks, common metrics include Mean Squared Error (MSE), Root Mean Squared "
        "Error (RMSE), Mean Absolute Error (MAE), and R-squared (coefficient of determination). "
        "RMSE is popular because it penalizes large errors more heavily and has the same units as "
        "the target variable.\n\n"
        "Cross-validation is a technique to assess model generalization. K-fold cross-validation "
        "splits the data into K subsets, trains on K-1 folds, and validates on the remaining fold. "
        "This process repeats K times, and results are averaged. It provides a more robust estimate "
        "of model performance than a single train-test split."
    ))

    # ── Page 5: Deployment and MLOps ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "Deployment and MLOps", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(0, 7, (
        "Deploying machine learning models to production involves several challenges beyond "
        "building accurate models. MLOps (Machine Learning Operations) is a set of practices "
        "that combines ML, DevOps, and data engineering.\n\n"
        "Model serving options include REST APIs (using Flask, FastAPI, or Streamlit), batch "
        "prediction pipelines, and edge deployment. The choice depends on latency requirements, "
        "scale, and infrastructure constraints.\n\n"
        "Model monitoring is essential in production. Data drift occurs when the statistical "
        "properties of input data change over time, causing model performance to degrade. Concept "
        "drift happens when the relationship between inputs and outputs changes. Both require "
        "continuous monitoring and periodic model retraining.\n\n"
        "Version control for ML extends beyond code to include data versioning (DVC), model "
        "versioning (MLflow), and experiment tracking. Reproducibility requires tracking not just "
        "the code but also the data, hyperparameters, and environment used for each experiment.\n\n"
        "Cloud platforms like AWS SageMaker, Google Vertex AI, and Azure ML provide end-to-end "
        "ML infrastructure. For simpler deployments, platforms like Render, Railway, and Hugging "
        "Face Spaces offer easy deployment options with free tiers suitable for personal projects "
        "and prototypes.\n\n"
        "Security considerations include protecting model endpoints, sanitizing inputs to prevent "
        "prompt injection, and ensuring sensitive training data isn't memorized or leaked by the "
        "model. API keys and credentials should never be hardcoded and should be managed through "
        "environment variables or secret management services."
    ))

    # Save the PDF
    output_path = "sample.pdf"
    pdf.output(output_path)
    print(f"✅ Created {output_path} ({pdf.pages_count} pages)")
    print(f"   Topics: ML Basics, Deep Learning, NLP, Evaluation, Deployment")
    print(f"\n   Now run: python step2_chunking.py")


if __name__ == "__main__":
    create_sample_pdf()
