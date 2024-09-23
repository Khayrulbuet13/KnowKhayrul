import os

from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI

# Models for handling QA and Cypher query generation
QA_MODEL = os.getenv("QA_MODEL")
CYPHER_MODEL = os.getenv("CYPHER_MODEL")

# Connect to Neo4j database
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

# Refresh schema to get the latest structure
graph.refresh_schema()

# Cypher query generation template
cypher_generation_template = """
Task:
Generate a Cypher query for a Neo4j graph database.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Schema:
{schema}

Note:
Do not include any explanations or apologies in your responses.
Do not respond to any questions that ask anything other than constructing a Cypher query.
Do not include any text except the generated Cypher statement.
Make sure the direction of the relationship is correct in your queries.
Ensure proper aliasing for entities and relationships. Do not run any queries that would add to or delete from
the database. Make sure to alias all statements that follow as `WITH` statements 
(e.g., WITH e AS education, c.CourseName AS CourseName).



Examples:
# What are the courses included in the Ph.D. program?
MATCH (e:Education {{degree: 'Ph.D. in Mechanical Engineering and Mechanics'}})-[:INCLUDED_IN]->(c:Course)
RETURN c.courseName AS course_name

# Which courses did the student/khayrul take in his M.S. degree?
MATCH (e:Education {{degree: 'M.S. in Mechanical Engineering and Mechanics'}})-[:INCLUDED_IN]->(c:Course)
RETURN c.courseName AS course_name

# How many courses were taken for the Ph.D. program?
MATCH (e:Education {{degree: 'Ph.D. in Mechanical Engineering and Mechanics'}})-[:INCLUDED_IN]->(c:Course)
RETURN COUNT(c) AS course_count

# What is Khayrul's GPA for a B.Sc.?
MATCH (e:Education {{degree: 'B.Sc. in Industrial and Production Engineering'}})
RETURN e.institution AS institution, e.gpa AS gpa
LIMIT 1

String category values:
Institutions are one of: 'Lehigh University', 'IBM', 'Bangladesh University of Engineering and Technology'
Locations are one of: 'Pennsylvania, USA', 'Online', 'Dhaka, Bangladesh'
Degrees are one of: 'Ph.D. in Mechanical Engineering and Mechanics', 'M.S. in Mechanical Engineering and Mechanics',
    'Professional Certificate in Data Science', 'B.Sc. in Industrial and Production Engineering'
Awards are one of:: 'P.C. Rossin College of Engineering fellowship', 'SCEA- PTAK prize global case study competition scholarship'




The question is:
{question}
"""

cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=cypher_generation_template
)

# QA generation template for interpreting Cypher results
qa_generation_template = """
You are an assistant that takes the results
from a Neo4j Cypher query and forms a human-readable response. The
query results section contains the results of a Cypher query that was
generated based on a users natural language question. The provided
information is authoritative, you must never doubt it or try to use
your internal knowledge to correct it. Make the answer sound like a
response to the question.

Query Results:
{context}

Question:
{question}

If the provided information is empty, say you don't know the answer.
Empty information looks like this: []

If the information is not empty, you must provide an answer using the
results. 

All the information you get thourgh query is about Khayrul's educational background.

Never say you don't have the right information if there is data in
the query results. Make sure to show all the relevant query results
if you're asked.

Helpful Answer:
"""

qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template
)

# Initialize the GraphCypherQAChain with the OpenAI model and Neo4j graph
hospital_cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(model=CYPHER_MODEL, temperature=0),
    qa_llm=ChatOpenAI(model=QA_MODEL, temperature=0),
    graph=graph,
    verbose=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    validate_cypher=True,
    top_k=100,
)
