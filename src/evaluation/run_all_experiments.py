import dagshub

# Dagshub must be initialized BEFORE mlflow import
dagshub.init(
    repo_owner="Nikhil-MLOPs",
    repo_name="Medwise-Rag",
    mlflow=True
)

import json
import time
import logging
from pathlib import Path
import mlflow

from src.evaluation.config import load_config
from src.evaluation.metrics import (
    recall,
    precision,
    citation_accuracy,
    extract_cited_pages,
    mean,
)
from src.retrieval.retriever import VectorRetriever
from src.rag.chain import RAGChain


# -------------------------------------------------------------------
# Logging configuration (NO loguru, stdlib logging only)
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("phase6.evaluation")


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_CONFIG_PATH = Path("configs/base.yaml")
EXPERIMENTS_DIR = Path("configs/experiments")
EVAL_QUESTIONS_PATH = Path("data/eval/questions.json")


def run_all_experiments():
    """
    Runs all config-driven RAG experiments and logs metrics to Dagshub MLflow.
    This function is intentionally verbose in logging to avoid silent blocking.
    """

    logger.info("Starting Phase 6: Evaluation & Controlled Experiments")

    mlflow.set_experiment("Evaluating - System")

    questions = json.loads(EVAL_QUESTIONS_PATH.read_text())
    logger.info(f"Loaded {len(questions)} evaluation questions")

    experiment_files = sorted(EXPERIMENTS_DIR.glob("exp_*.yaml"))
    logger.info(f"Discovered {len(experiment_files)} experiment configs")

    for exp_index, exp_path in enumerate(experiment_files, start=1):
        logger.info("=" * 80)
        logger.info(f"[{exp_index}/{len(experiment_files)}] Starting experiment: {exp_path.name}")

        config = load_config(BASE_CONFIG_PATH, exp_path)

        with mlflow.start_run(run_name=exp_path.stem):
            mlflow.log_params(config["chunking"])
            mlflow.log_params(config["retrieval"])
            mlflow.log_params(config["llm"])

            logger.info("Initializing VectorRetriever")
            retriever = VectorRetriever(
                vectorstore_dir=Path("data/vectorstore"),
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                k=config["retrieval"]["k"],
                strategy=config["retrieval"]["strategy"],
            )

            logger.info("Initializing RAGChain (non-streaming for evaluation)")
            rag_chain = RAGChain(retriever)

            recalls, precisions, citations = [], [], []
            retrieval_latencies, generation_latencies, e2e_latencies = [], [], []

            for q_index, item in enumerate(questions, start=1):
                question = item["question"]
                expected_pages = item["expected_pages"]

                logger.info(f"Experiment {exp_path.stem} | Question {q_index}/{len(questions)}")
                logger.info(f"Question text: {question}")

                t_start = time.perf_counter()

                logger.info("Running retrieval step")
                t_retrieval_start = time.perf_counter()
                documents = retriever.retrieve(question)
                t_retrieval_end = time.perf_counter()

                logger.info(
                    f"Retrieved {len(documents)} documents in "
                    f"{t_retrieval_end - t_retrieval_start:.2f}s"
                )

                logger.info("Running LLM generation step")
                t_generation_start = time.perf_counter()
                answer = rag_chain.generate_answer(question)
                t_generation_end = time.perf_counter()

                logger.info(
                    f"Generated answer in "
                    f"{t_generation_end - t_generation_start:.2f}s"
                )

                t_end = time.perf_counter()

                retrieved_pages = [
                    doc.metadata.get("page")
                    for doc in documents
                    if doc.metadata.get("page") is not None
                ]
                cited_pages = extract_cited_pages(answer)

                recalls.append(recall(retrieved_pages, expected_pages))
                precisions.append(precision(retrieved_pages, expected_pages))
                citations.append(citation_accuracy(cited_pages, retrieved_pages))

                retrieval_latencies.append(t_retrieval_end - t_retrieval_start)
                generation_latencies.append(t_generation_end - t_generation_start)
                e2e_latencies.append(t_end - t_start)

                logger.info(
                    f"Metrics so far | recall={recalls[-1]:.2f}, "
                    f"precision={precisions[-1]:.2f}, "
                    f"citation_accuracy={citations[-1]:.2f}"
                )

            logger.info(f"Logging aggregated metrics for {exp_path.stem}")

            mlflow.log_metric("recall", mean(recalls))
            mlflow.log_metric("precision", mean(precisions))
            mlflow.log_metric("citation_accuracy", mean(citations))
            mlflow.log_metric("retrieval_latency", mean(retrieval_latencies))
            mlflow.log_metric("generation_latency", mean(generation_latencies))
            mlflow.log_metric("end_to_end_latency", mean(e2e_latencies))

            logger.info(f"Completed experiment: {exp_path.stem}")

    logger.info("All experiments completed successfully")


if __name__ == "__main__":
    run_all_experiments()
