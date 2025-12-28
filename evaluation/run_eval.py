import os
import sys
import time
import logging
from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextUtilization

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EvalPipeline")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rag import engine

class RAGEvaluator:
    def __init__(self, questions):
        self.questions = questions
        self.metrics = [Faithfulness(), AnswerRelevancy(), ContextUtilization()]

    def _safe_generate(self, chain, question):
        """Метод з обробкою помилок та ретраями (Retry logic)"""
        for attempt in range(3):
            try:
                return chain.invoke(question)
            except Exception as e:
                wait_time = (attempt + 1) * 5
                logger.warning(f"Rate limit hit. Waiting {wait_time}s... Error: {e}")
                time.sleep(wait_time)
        return "I don't know (API Failure)"

    def generate_dataset(self):
        answers = []
        contexts = []
        chain = engine.get_chain()
        
        logger.info("Starting Data Generation Phase")
        for q in self.questions:
            docs = engine.vector_store.as_retriever(search_kwargs={"k": 1}).invoke(q)
            
            content = [d.page_content[:1500].replace('\x00', '') for d in docs]
            contexts.append(content)
            
            ans = self._safe_generate(chain, q)
            answers.append(ans)
            logger.info(f"Generated answer for: {q[:30]}...")
            
        return Dataset.from_dict({
            "question": self.questions,
            "answer": answers,
            "contexts": contexts
        })

    def run(self):
            dataset = self.generate_dataset()
            
            # Видаляємо thread_timeout, залишаємо базові параметри
            run_config = RunConfig(
                max_workers=1, 
                timeout=300
            )

            logger.info("Starting Ragas Evaluation Phase")
            try:
                results = evaluate(
                    dataset=dataset,
                    metrics=self.metrics,
                    llm=engine.llm,
                    embeddings=engine.embeddings,
                    run_config=run_config,
                    raise_exceptions=False
                )
                self.log_and_save(results)
            except Exception as e:
                logger.error(f"Evaluation failed: {e}")

    def log_and_save(self, results):
        os.makedirs("evaluation", exist_ok=True)
        results.to_pandas().to_csv("evaluation/report.csv", index=False)
        logger.info("\n" + "="*20 + "\nEVALUATION COMPLETE\n" + str(results) + "\n" + "="*20)

if __name__ == "__main__":
    QUESTIONS = [
        "Who initiated the project that led to PDF?",
        "What is the ISO standard number for PDF?",
        "On which language is the PDF structure based?"
    ]
    evaluator = RAGEvaluator(QUESTIONS)
    evaluator.run()