import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from ml_utils import load_model, predict, explainability#, retrain
from typing import List
import sys
from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles
sys.setrecursionlimit(10000)



# defining the main app
app = FastAPI(title="Bug Predictor", docs_url="/")

# calling the load_model during startup.
# this will train the model and keep it loaded for prediction.
app.add_event_handler("startup", load_model)

# app.mount("/dataset", StaticFiles(directory="static", html = True), name="static")



# class which is expected in the payload
class QueryIn(BaseModel):
    line_count_of_code: float
    cyclomatic_complexity: float
    essential_complexity: float
    design_complexity: float
    total_operators_operands: float
    volume: float
    program_length: float
    difficulty: float
    intelligence: float
    effort: float
    b: float
    time_estimator: float
    line_count: float
    count_of_lines_of_comments: float
    count_of_blank_lines: float
    count_of_CodeAndComment: float
    unique_operators: float
    unique_operands: float
    total_operators: float
    total_operands: float
    branchCount_of_flow_graph: float


# class which is returned in the response
class QueryOut(BaseModel):
    defect: bool

# class which is expected in the payload while re-training
# class FeedbackIn(BaseModel):
#     sepal_length: float
#     sepal_width: float
#     petal_length: float
#     petal_width: float
#     flower_class: str

# Route definitions
@app.get("/ping")
# Healthcheck route to ensure that the API is up and running
def ping():
    return {"ping": "pong"}


@app.post("/predict_bug", response_model=QueryOut, status_code=200)
# Route to do the prediction using the ML model defined.
# Payload: QueryIn containing the parameters
# Response: QueryOut containing the flower_class predicted (200)
def predict_flower(query_data: QueryIn):
    output = {"defect": predict(query_data)}
    return output

@app.get("/explain", status_code=200)
# Healthcheck route to ensure that the API is up and running
async def explain():
    # exm = explainability()
    some_file_path = "dataset/explainable_AI_starter.html"
    return FileResponse(some_file_path, filename="explain.html")

# @app.post("/feedback_loop", status_code=200)
# # Route to further train the model based on user input in form of feedback loop
# # Payload: FeedbackIn containing the parameters and correct flower class
# # Response: Dict with detail confirming success (200)
# def feedback_loop(data: List[FeedbackIn]):
#     retrain(data)
#     return {"detail": "Feedback loop successful"}


# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="0.0.0.0", port=8889, reload=True)
