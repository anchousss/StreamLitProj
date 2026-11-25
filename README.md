# Fashion-MNIST Web Service

Веб-сервис классификации одежды на базе Fashion-MNIST.

## Стек
- PyTorch (CNN)
- FastAPI (REST API)
- Streamlit (интерфейс)
- Docker / Docker Compose

## Быстрый старт (локально)

```bash
pip install -r requirements.txt
python src/train.py 10          # ~10-15 мин на CPU, быстрее на GPU
uvicorn api.main:app --reload   # терминал 1
streamlit run app/streamlit_app.py  # терминал 2