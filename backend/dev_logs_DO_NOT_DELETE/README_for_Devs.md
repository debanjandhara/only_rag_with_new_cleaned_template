# Vectorization Development Logs

## Initialization
- The vectorDB is initialized using a sample file `initialize.txt`.
- This setup simplifies merging and deletion functions.

## Merging
- **Merge Function**: Adds new vectorDB (created when adding new documents) to the existing one.

## Deletion
- **Delete Function**: 
  - No direct delete function available; use a workaround with pandas.
  - Can be used even if the file is not present, for safety.
  - **Caution**: 
    - The directory/filename of chunks in vectorDB behave differently in the deployed version.
    - There are two delete functions: check the vectorstore in the deployed version via log (`output.csv`) and act accordingly.

## Extra Tasks
1. **VectorDB Update**:
   - **Function**: `load_vectorstore_and_retriever()` reloads the vectorDB in a variable to update the context.
   - **Issue**: The function doesn't work as expected. 
   - **Workaround**: Close and restart the Python FastAPI service after adding/deleting entries in vectorDB. This seems to work for now but needs a resolution.
2. **File Extension Check**:
   - Ensure file extensions match predefined formats before uploading.
   - If extensions don't match, prompt users to upload files in predefined formats only.
