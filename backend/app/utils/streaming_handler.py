import asyncio

from app.utils.logger import log_error
from app.utils.openai_call_functions import invoke_rag_chain

async def handle_request(query, session_id):
    try:
        print("Handling request for session:", session_id)
        responses = []
        async for response in invoke_rag_chain(query, session_id):
            responses.append(response)
        print("AI response received")
        return responses
    except Exception as e:
        error_id = "ERR001"
        log_error(error_id, "handle_request", str(e))
        print(f"Error {error_id} occurred in handle_request: {str(e)}")
        return None

async def stream_response_chunks(responses):
    try:
        for response in responses:
            if 'answer' in response:
                yield response['answer']
                await asyncio.sleep(0.01)  # Adjust the sleep time as needed
    except Exception as e:
        error_id = "ERR002"
        log_error(error_id, "stream_response_chunks", str(e))
        print(f"Error {error_id} occurred in stream_response_chunks: {str(e)}")
        yield f"Error {error_id}: {str(e)}"
