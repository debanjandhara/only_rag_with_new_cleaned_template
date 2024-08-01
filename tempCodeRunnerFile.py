@app.post("/stream")
async def stream_response(request_body: QueryRequest):
    try:
        print(f"Received query: {request_body.query} with session_id: {request_body.session_id} and history_id: {request_body.history_id}")

        if not request_body.query or not request_body.session_id or not request_body.history_id:
            error_message = "All 'query', 'session_id', and 'history_id' must be provided"
            print(error_message)
            return JSONResponse(content={"error": error_message}, status_code=400)

        responses = await handle_request(request_body.session_id, request_body.query, request_body.history_id)
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