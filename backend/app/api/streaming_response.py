from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates

from app.models.pydantic_models import QueryRequest
from app.utils.streaming_handler import handle_request, stream_response_chunks

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/chatbot", response_class=HTMLResponse)
async def get_chatbot(request: Request):
    return templates.TemplateResponse("chatbot.html", {"request": request})

@router.post("/stream")
async def stream_response(request_body: QueryRequest):
    try:
        print(f"Received query: {request_body.query} and session_id: {request_body.session_id}")

        if not request_body.query or not request_body.session_id:
            error_message = "All 'query' and 'session_id' must be provided"
            print(error_message)
            return JSONResponse(content={"error": error_message}, status_code=400)

        responses = await handle_request(request_body.query, request_body.session_id)
        if responses is None:
            error_message = "Error processing the request"
            print(error_message)
            return JSONResponse(content={"error": error_message}, status_code=500)

        return StreamingResponse(stream_response_chunks(responses), media_type="text/plain")
    except HTTPException as http_err:
        error_id = "ERR010"
        error_message = f"HTTPException {error_id}: {str(http_err)}"
        print(error_message)
        return JSONResponse(content={"error": error_message}, status_code=http_err.status_code)
    except Exception as e:
        error_id = "ERR010"
        error_message = f"Error {error_id}: {str(e)}"
        print(error_message)
        return JSONResponse(content={"error": error_message}, status_code=500)