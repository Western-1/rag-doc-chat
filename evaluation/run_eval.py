import os
import sys
import time
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextUtilization
from src.config import COLLECTION_NAME

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rag import engine

class RAGEvaluator:
    def __init__(self, questions):
        self.questions = questions
        self.metrics = [Faithfulness(), AnswerRelevancy(), ContextUtilization()]

    def generate_dataset(self):
        answers, contexts, chain = [], [], engine.get_chain()
        print("🚀 Generating responses (Safe Mode: 10s delay)...")
        
        for i, q in enumerate(self.questions):
            print(f"[{i+1}/{len(self.questions)}] Processing: {q[:30]}...")
            
            docs = engine.vector_store.as_retriever(search_kwargs={"k": 3}).invoke(q)
            contexts.append([d.page_content[:2000] for d in docs])
            
            try:
                ans = chain.invoke(q)
                answers.append(ans)
            except Exception as e:
                print(f"❌ Error: {e}")
                answers.append("API Failure")
            
            time.sleep(10)
            
        return Dataset.from_dict({
            "question": self.questions, 
            "answer": answers, 
            "contexts": contexts
        })

    def run(self):
        try:
            count = engine.client.count(COLLECTION_NAME).count
            if count == 0:
                print("⚠️ Qdrant empty. Skipping.")
                return
        except Exception:
            print("⚠️ Qdrant connection failed. Skipping.")
            return

        dataset = self.generate_dataset()
        
        print("🚀 Running Ragas evaluation...")
        results = evaluate(
            dataset=dataset, 
            metrics=self.metrics, 
            llm=engine.llm, 
            embeddings=engine.embeddings, 
            run_config=RunConfig(max_workers=1, timeout=300)
        )
        
        os.makedirs("evaluation", exist_ok=True)
        results.to_pandas().to_csv("evaluation/report.csv", index=False)
        print(f"\n✅ Final Metrics:\n{results}")

if __name__ == "__main__":
    qs = [
        "Who initiated the project that led to PDF?", 
        "What is the ISO standard number for PDF?", 
        "On which language is the PDF structure based?"
    ]
    RAGEvaluator(qs).run()