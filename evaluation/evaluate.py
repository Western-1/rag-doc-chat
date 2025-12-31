import os
import sys
import logging
import time

# --- ФІКС ШУМУ ---
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("groq").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

import pandas as pd
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import Faithfulness, ContextPrecision, AnswerSimilarity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rag import engine
from src.config import COLLECTION_NAME

class RAGEvaluator:
    def __init__(self, questions, ground_truths=None):
        self.questions = questions
        self.ground_truths = ground_truths
        
        self.metrics = [
            Faithfulness(),
            ContextPrecision(),
            AnswerSimilarity()
        ]

    def generate_dataset(self):
        answers = []
        contexts = []
        
        print(f"🚀 Starting evaluation on {len(self.questions)} questions...")
        
        for i, q in enumerate(self.questions):
            print(f"[{i+1}/{len(self.questions)}] Asking RAG: {q}")
            try:
                ans, source_docs, _ = engine.get_answer_with_sources(q)
                
                current_contexts = []
                for doc in source_docs:
                    if isinstance(doc, dict):
                        current_contexts.append(doc.get('text', ''))
                    else:
                        current_contexts.append(doc.page_content)

                answers.append(ans)
                contexts.append(current_contexts)
                
                time.sleep(1) 
                
            except Exception as e:
                print(f"❌ Error processing '{q}': {e}")
                if len(answers) == len(contexts):
                    answers.append("Error")
                    contexts.append(["Error"])
                elif len(answers) > len(contexts):
                    contexts.append(["Error"])

        data_dict = {
            "question": self.questions,
            "answer": answers,
            "contexts": contexts
        }
        
        if self.ground_truths:
            data_dict["ground_truth"] = self.ground_truths

        return Dataset.from_dict(data_dict)

    def run(self):
        try:
            count = engine.client.count(COLLECTION_NAME).count
            if count == 0:
                print("⚠️ Qdrant empty.")
                return
        except Exception:
            return

        dataset = self.generate_dataset()
        
        print("\n📊 Calculating Metrics...")
        results = evaluate(
            dataset=dataset, 
            metrics=self.metrics, 
            llm=engine.llm, 
            embeddings=engine.embeddings,
            run_config=RunConfig(max_workers=1, timeout=120, max_retries=3),
            raise_exceptions=False 
        )
        
        os.makedirs("evaluation", exist_ok=True)
        results.to_pandas().to_csv("evaluation/report.csv", index=False)
        
        print("\n✅ Final Report Saved to 'evaluation/report.csv'")
        print(results)
        return results

if __name__ == "__main__":
    qs = [
        "Who initiated the project that led to PDF?", 
        "What is the ISO standard number for PDF?", 
        "On which language is the PDF structure based?"
    ]
    gt = [
        "Dr. John Warnock",
        "ISO 32000",
        "PostScript programming language"
    ]
    RAGEvaluator(qs, gt).run()