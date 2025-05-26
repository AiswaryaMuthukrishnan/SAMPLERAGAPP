from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from rag_engine import RAGEngine
import os

app = FastAPI()
rag = RAGEngine()

UPLOAD_DIR = "storage"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_pdf(file: UploadFile):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        f.write(await file.read())

    chunks = rag.process_pdf(filepath)
    rag.build_index(chunks)
    return {"status": "PDF processed and indexed."}

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    if not rag.index:
        return JSONResponse(status_code=400, content={"error": "No document indexed yet."})
    answer = rag.query(question)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
