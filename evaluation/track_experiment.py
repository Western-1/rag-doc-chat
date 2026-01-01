import os
import sys
import time
import logging
import wandb
from datasets import Dataset, Features, Sequence, Value
from ragas import evaluate, RunConfig
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# 1. Приглушуємо шумні бібліотеки
for logger_name in ["httpx", "httpcore", "groq", "openai", "qdrant_client", "sentence_transformers"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Додаємо корінь проєкту в шлях для імпорту src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.rag import engine as rag_engine
from src.config import CHUNK_SIZE, LLM_MODEL

# Налаштування логера
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("WANDB-EVAL")

class RAGWandbEvaluator:
    def __init__(self):
        # Тестові дані, які точно співпадають із нашим згенерованим PDF
        self.eval_data = [
            {
                "question": "What is the main topic of this document?", 
                "ground_truth": "The document discusses RAG architecture and MLOps integration."
            },
            {
                "question": "Which embedding model is used?", 
                "ground_truth": "The system uses huggingface/all-MiniLM-L6-v2."
            }
        ]
        self.metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        self.test_file = "test_data_autogen.pdf"

    def _create_dummy_pdf(self):
        """
        Створює PDF файл програмно. 
        Це замінює необхідність ручного створення example.pdf.
        """
        text_content = """
        Talk to Your Docs System.
        This document discusses RAG architecture and MLOps integration.
        The system uses huggingface/all-MiniLM-L6-v2 embedding model for semantic search.
        """
        
        try:
            # Спроба використати reportlab для "чесного" PDF
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(self.test_file)
            c.drawString(100, 750, "RAG MLOps Test Document")
            y = 700
            for line in text_content.split('\n'):
                c.drawString(50, y, line.strip())
                y -= 20
            c.save()
            logger.info(f"✅ Generated synthetic PDF using ReportLab: {self.test_file}")
            
        except ImportError:
            # Фолбек: Створюємо PDF "хардкорно" (простий формат, який зрозуміє PyPDFLoader)
            # Якщо reportlab не встановлено, ми все одно не впадемо.
            logger.warning("⚠️ ReportLab not found. Creating simple text-based PDF fallback.")
            with open(self.test_file, "w") as f:
                # Це не валідний бінарний PDF, але PyPDFLoader іноді може читати текст
                # Краще рішення для продакшну: `pip install reportlab`
                f.write(text_content)

    def _ensure_data_exists(self):
        """
        Перевіряє базу. Якщо пуста -> генерує файл -> завантажує -> видаляє файл.
        """
        try:
            # Перевіряємо, чи є документи в колекції
            info = rag_engine.client.get_collection(rag_engine.vector_store.collection_name)
            if info.points_count == 0:
                logger.warning("⚠️ Database is empty! Starting auto-ingestion...")
                
                # 1. Створюємо файл
                self._create_dummy_pdf()
                
                # 2. Завантажуємо в RAG
                if os.path.exists(self.test_file):
                    rag_engine.ingest_file(self.test_file)
                    logger.info(f"✅ Ingested {self.test_file} into Qdrant")
                    
                    # 3. Прибираємо за собою (Cleanup)
                    os.remove(self.test_file)
                    logger.info(f"🧹 Cleaned up temporary file: {self.test_file}")
                else:
                    logger.error("❌ Failed to create test file.")
            else:
                logger.info("✅ Database already has data. Skipping ingestion.")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not check DB status (might be connection error): {e}")

    def _generate_dataset(self):
        questions, answers, contexts, ground_truths = [], [], [], []
        
        logger.info(f"🚀 Starting RAG inference on {len(self.eval_data)} samples...")
        
        for i, item in enumerate(self.eval_data):
            q = item["question"]
            logger.info(f"[{i+1}/{len(self.eval_data)}] Processing: {q}...")
            
            try:
                ans, sources, _ = rag_engine.get_answer_with_sources(query=q)
                
                questions.append(q)
                answers.append(ans)
                # Витягуємо текст з джерел
                contexts.append([s['text'] for s in sources])
                ground_truths.append(item["ground_truth"])
                
                # Пауза для Groq Free Tier
                time.sleep(1.5) 
                
            except Exception as e:
                logger.error(f"❌ Error at sample {i+1}: {e}")

        # Сувора схема даних для Arrow/Ragas
        features = Features({
            'question': Value('string'),
            'answer': Value('string'),
            'contexts': Sequence(Value('string')),
            'ground_truth': Value('string')
        })
        
        return Dataset.from_dict({
            "question": questions, "answer": answers, 
            "contexts": contexts, "ground_truth": ground_truths
        }, features=features)

    def run(self):
        # 1. Підготовка даних (Cold Start)
        self._ensure_data_exists()

        # 2. Ініціалізація W&B
        run = wandb.init(
            project="talk-to-your-docs-rag",
            name=f"eval-{LLM_MODEL.replace('/', '-')}-v3",
            config={"chunk_size": CHUNK_SIZE, "llm": LLM_MODEL}
        )

        # 3. Генерація датасету
        dataset = self._generate_dataset()

        logger.info("\n📊 Calculating Ragas Metrics (LLM-as-a-Judge)...")
        # max_workers=1 важливо для уникнення лімітів API
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=rag_engine.llm,
            embeddings=rag_engine.embeddings,
            run_config=RunConfig(max_workers=1, timeout=60, max_retries=2)
        )

        # 4. Логування результатів
        wandb.log(results)

        # Логування таблиці для ручного аналізу
        eval_df = results.to_pandas()
        wandb.log({"detailed_results": wandb.Table(dataframe=eval_df)})

        logger.info(f"\n✅ Evaluation Complete. Results:\n{results}")
        wandb.finish()

if __name__ == "__main__":
    evaluator = RAGWandbEvaluator()
    evaluator.run()