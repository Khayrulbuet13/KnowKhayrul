import logging
import os
from neo4j import GraphDatabase
from retry import retry

# Environment variables
# Set the environment variables directly in the script for clarity and testing purposes
os.environ['EDUCATION_CSV_PATH'] = 'https://raw.githubusercontent.com/Khayrulbuet13/KnowKhayrul/main/data/education.csv'
os.environ['COURSES_CSV_PATH'] = 'https://raw.githubusercontent.com/Khayrulbuet13/KnowKhayrul/main/data/courses.csv'

EDUCATION_CSV_PATH = os.getenv("EDUCATION_CSV_PATH")
COURSES_CSV_PATH = os.getenv("COURSES_CSV_PATH")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
LOGGER = logging.getLogger(__name__)

NODES = ["Education", "Course"]

def set_uniqueness_constraints(session, node):
    """Creates a uniqueness constraint for a node on the 'id' property."""
    query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{node}) REQUIRE n.id IS UNIQUE"
    session.run(query)
    LOGGER.info(f"Uniqueness constraint ensured for {node} on id.")

@retry(tries=100, delay=10)
def load_educational_data():
    """Loads educational data into Neo4j from specified CSV paths."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session(database="neo4j") as session:
        # Setting uniqueness constraints on all nodes
        for node in NODES:
            set_uniqueness_constraints(session, node)

        # Loading data from education CSV
        query_education = f"""
            LOAD CSV WITH HEADERS FROM '{EDUCATION_CSV_PATH}' AS row
            MERGE (e:Education {{
                id: row.Education_ID,
                institution: row.Institution,
                location: row.Location,
                degree: row.Degree,
                gpa: COALESCE(row.GPA, 'Not Available'),
                startDate: COALESCE(row.StartDate, 'Date Not Provided'),
                endDate: COALESCE(row.EndDate, 'Date Not Provided'),
                awards: COALESCE(row.Awards, 'None'),
                thesisTitle: COALESCE(row.ThesisTitle, 'Not Provided')
            }});
        """
        session.run(query_education)
        LOGGER.info("Education data loaded successfully.")

        # Loading data from courses CSV
        query_courses = f"""
            LOAD CSV WITH HEADERS FROM '{COURSES_CSV_PATH}' AS row
            MERGE (c:Course {{
                id: row.ID,
                EducationID: row.Education_ID,
                courseID: row.CourseID,
                courseName: row.CourseName,
                courseGrade: row.CourseGrade
            }});
        """
        session.run(query_courses)
        LOGGER.info("Course data loaded successfully.")


        # Creating relationships
        query_relationships = f"""
            LOAD CSV WITH HEADERS FROM '{COURSES_CSV_PATH}' AS courses
                MATCH (e:Education {{id: courses.Education_ID}})
                MATCH (c:Course {{id: courses.ID}})
                MERGE (e)-[:INCLUDED_IN]->(c)
            """
        session.run(query_relationships)
        LOGGER.info("Relationships between Education and Courses established.")




if __name__ == "__main__":
    load_educational_data()
