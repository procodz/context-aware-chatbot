# system_promt = """
# You explain things to people like they are five years old.
# """
# user_prompt = """
# What is Langchain?
# """

# messages = [
#     SystemMessage(content=system_promt),
#     HumanMessage(content=user_prompt)
# ]


# Get response from the model
# response = llm_gpt4.invoke(messages)

# answer = textwrap.fill(response.content, width=100)
# print(answer)
# promt_template = """
# you are a helppful assistant that explains AI topics. Given the following input:
# {topic} 
# Provide an explanation of given topic.
# """

# prompt = PromptTemplate(
#     template=promt_template,
#     input_variables=["topic"]
# )
# chain = promt | llm_gpt4
# chain.invoke({"topic": "what isLangchain"}).content

# loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=8BV9TW490nQ&list=PLHZWvoOoCcSzGmTtfHEkaFQDIJuCD8vDR&index=5&ab_channel=Rabbitmetrics", add_video_info=False )

# docs = loader.load()
# transcript = docs[0].page_content
# summarize_promt_template = """
# you are a helppful assistant that explains AI topics. Given the following input:
# {context} 
# summarise the context
# """

# prompt = PromptTemplate(
#     template=promt_template,
#     input_variables=["context"]
# )
# summarize_promt = PromptTemplate.from_template(summarize_promt_template)
# output_parser = StrOutputParser()
# summarize_chain = summarize_promt | llm_gpt4 | output_parser #example runnable sequence
# # summarize_chain.invoke({"context": "what is langchain?"})
# length_lambda = RunnableLambda(lambda summary: f"summary length: {len(summary)} characters")

# lambda_chain = summarize_chain | length_lambda

# lambda_chain.invoke({"context": "what is langchain?"})
# # chain = create_stuff_documents_chain(llm_gpt4, prompt)
# # chain = prompt | llm_gpt4
# # print(type(lambda_chain.steps[-1]))
# print(res)

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 100,
#     chunk_overlap  = 20,
#     length_function = len,
#     is_separator_regex = False
# )

# docs_split = text_splitter.split_documents(docs)

# r = redis.Redis(
#     host = REDIS_HOST,
#     port = REDIS_PORT,
#     password = REDIS_PASSWORD
# )
# embeddings = HuggingFaceBgeEmbeddings();

# rds = Redis.from_documents(
#     docs_split,
#     embeddings,
#     redis_url = REDIS_URL,
#     index_name = "youtube"
# )

# # print(rds.index_name)
# retriever = rds.as_retriever(
#     search_type = "similarity",
#     search_kwargs = {
#         "k": 10
#     }
# )

# # print(retriever.invoke("data analysis"))

# template = """
# Answer the question based only on the following context:
# {context}

# Question: {question}
# """
# prompt = PromptTemplate.from_template(template)
# output_parser = StrOutputParser()

# chain = (
#     {"context": (lambda x: x["question"]) | retriever,
#     "question": (lambda x: x["question"])}
#     | prompt
#     | llm_gpt4
#     | output_parser
# )

# answer = chain.invoke({"question": "what can you do with LLama 3?"})
# print(answer)

# youtube_tool = YouTubeSearchTool()

# llm_with_tools = llm_gpt4.bind_tools([youtube_tool])

# msg = llm_with_tools.invoke("find some rabbitmatrics videos on langchain")
# print(msg.tool_calls)
# chain = llm_with_tools | (lambda x: x.tool_calls[0]["args"]["query"]) | youtube_tool

# print(chain.invoke("find some rabbitmatrics videos on langchain"))

# prompt = hub.pull("hwchase17/openai-tools-agent")
# # print(promt.messages)

# tools = [youtube_tool]

# agent = create_tool_calling_agent(
#     llm_gpt4,
#     tools,
#     prompt
# )

# agent_executor = AgentExecutor(agent = agent, tools = tools, verbose = True)

# res = agent_executor.invoke({
#     "input": "find some rabbitmatrics videos on langchain"
# })
# print(res)