> # Text Splitting 

The process of breaking large text into multiple smaller, manageable pieces that an LLM can handle effectively 

## Why chunking
1. Embedding and semantic meaning capturing: A large single chunk of text is not ideal for vector representation, as it makes it harder to capture precise semantic meaning. Chunking the text into smaller segments and generating a vector representation for each chunk results in much more effective and meaningful embeddings.
2. Semantic search: Chunking make it possible to more accurately search in the vector.
3. Summarization: Bigger text are more prone to hallucination. To prevent chunking work as an effective solution.
4. Chunking is a more memory-efficient method and allow for better **parallelization**. It also require less computational resources. 

## Types of text splitter 
1. Length based 
2. Text structure based 
3. Document structure based 
4. Semantic meaning based 


