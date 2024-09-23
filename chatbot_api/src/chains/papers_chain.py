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

# Cypher generation template tailored for Papers and Skills
cypher_generation_template = """
Task:
Generate a Cypher query for a Neo4j graph database based on the provided schema and user question.

Instructions:
- Use only the relationship types and properties defined in the schema below.
- Do not introduce any new relationship types or properties.
- Do not include explanations, apologies, or any text outside the Cypher query.
- Ensure the direction of relationships is correct.
- Use proper aliasing for entities and relationships.
- Do not perform any operations that modify the database (e.g., CREATE, DELETE).
- Alias all intermediate statements using `WITH` clauses as necessary.

Schema:
{schema}

Nodes:
- Paper(id, title, abstract_novelty, abstract_challenge, abstract_result, keywords, issue, author, date, doi, journaltitle, pages, volume, contribution, first_author, publisher, url)
- Skill(id, skill, skill_type)

Relationships:
- (Paper)-[:UTILIZES]->(Skill)

Example Questions and Cypher Queries:
# What are the papers published by Khayrul as the first author?
MATCH (p:Paper)
WHERE p.first_author = 'True' AND toLower(p.author) CONTAINS toLower('Khayrul')
RETURN p.title AS paper_title

# What are the papers Khayrul published?
MATCH (p:Paper)
WHERE toLower(p.author) CONTAINS toLower('Khayrul')
RETURN p.title AS paper_title

# List all skills associated with my papers.
MATCH (p:Paper)-[:UTILIZES]->(s:Skill)
RETURN DISTINCT s.skill AS skill_name

# How many papers did Khayrul publish during PhD?
MATCH (p:Paper)
WHERE p.date >= '2021-01-01' AND toLower(p.author) CONTAINS toLower('Khayrul')
RETURN COUNT(p) AS paper_count

# How many papers did Khayrul publish during MS?
MATCH (p:Paper)
WHERE p.date >= '2021-01-01' AND p.date < '2023-06-17' AND toLower(p.author) CONTAINS toLower('Khayrul')
RETURN COUNT(p) AS paper_count

# How many papers did Khayrul publish during BSc?
MATCH (p:Paper)
WHERE p.date < '2021-01-01' AND toLower(p.author) CONTAINS toLower('Khayrul')
RETURN COUNT(p) AS paper_count



# What software was used in the paper titled "Tailoring polyamide nanocomposites: The synergistic effects of SWCNT chirality and maleic anhydride grafting"?
MATCH (p:Paper {{title: 'Tailoring polyamide nanocomposites: The synergistic effects of SWCNT chirality and maleic anhydride grafting'}})-[:UTILIZES]->(s:Skill)
WHERE s.skill_type = 'Design and Simulation Software'
RETURN s.skill AS software_used

# List all papers published in the journal "ACS Applied Engineering Materials".
MATCH (p:Paper)
WHERE p.journaltitle = 'ACS Applied Engineering Materials'
RETURN p.title AS paper_title, p.date AS publication_date

# What is the novelty of the paper titled "MIML: Multiplex Image Machine Learning for High Precision Cell Classification via Mechanical Traits within Microfluidic Systems"?
MATCH (p:Paper {{title: 'MIML: Multiplex Image Machine Learning for High Precision Cell Classification via Mechanical Traits within Microfluidic Systems'}})
RETURN p.abstract_novelty AS novelty

# Which programming languages do you know?
MATCH (s:Skill {{skill_type: 'Programming Languages'}})
RETURN s.skill AS skill_name



The question is:
{question}

"""

cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=cypher_generation_template
)

# QA generation template for interpreting Cypher results related to Papers and Skills
qa_generation_template = """
You are an assistant that takes the results
from a Neo4j Cypher query and forms a human-readable response. The
query results section contains the results of a Cypher query that was
generated based on a user's natural language question. The provided
information is authoritative; you must never doubt it or try to use
your internal knowledge to correct it. Make the answer sound like a
response to the question.

Query Results:
{context}

Question:
{question}

Guidelines:
- If the provided information is empty (e.g., []), respond with: "I don't have the information to answer that question."
- If the information is not empty, provide a clear and concise answer using the results.
- All information pertains to your published papers and the skills utilized or gained.
- Never state that you lack information if query results are present.
- Never include node that are not related to the query.
- Include all relevant query results in your response if applicable.
- Md Khayrul Islam is the person whose published paper and skills is being queried. Any of the following can refer to him: 'Md Khayrul Islam', 'Md', 'Khayrul', 'Islam', 'Khayrul Islam', 'Md Khayrul', or 'Islam'. Additionally, 'Mr./Dr. Islam', 'Mr./Dr. Khayrul', or 'Mr./Dr. Khayrul Islam' can also be used.
- skills types are Programming Languages,  Tools and Libraries, Design and Simulation Software, Data Analysis and Machine Learning, Soft Skills.
- When the user's question refers to educational levels (e.g., PhD, MS, BSc), map them to the following date ranges:
  - PhD: Papers published after '2021-01-01' (`p.date >= '2021-01-01'`)
  - MS: Papers published between '2021-01-01' and '2023-06-17' (`p.date >= '2021-01-01' AND p.date < '2023-06-17'`)
  - BSc: Papers published before '2021-01-01' (`p.date < '2021-01-01'`)


Helpful Answer:
"""

qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template
)

# Initialize the GraphCypherQAChain with the OpenAI model and Neo4j graph
papers_chain = GraphCypherQAChain.from_llm(
    cypher_llm=ChatOpenAI(model=CYPHER_MODEL, temperature=0),
    qa_llm=ChatOpenAI(model=QA_MODEL, temperature=0),
    graph=graph,
    verbose=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    validate_cypher=True,
    top_k=100,
)
