Vectorisation :

there is a merge function , we use that to add new vectorDB to the existing one

there are no such direct delete fucntion, we have to find a work around using pandas

delete function --> can be used even if, the file is not present... to be safe

Caution : the directory / filename of the chunks in vectorDB behaves differently in deployed version, so there are 2 function of delete... view the vectorstore in the deployed version via log, and act accordinly

> Tasks for DD : 

whenever you're updating something in VectorDB, make sure to load them again....