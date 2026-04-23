import uuid

from neo4j import GraphDatabase

# with this
try:
    from resume_parser.config import NEO_CONNECTION_URI, NEO_PASSWORD, NEO_USERNAME
except ImportError:
    from config import NEO_CONNECTION_URI, NEO_PASSWORD, NEO_USERNAME
try:
    from resume_parser.schemas import ParsedResume
except ImportError:
    from schemas import ParsedResume


def get_neo4j_driver():
    if not NEO_CONNECTION_URI or not NEO_USERNAME or not NEO_PASSWORD:
        raise ValueError(
            "Neo4j env vars missing. Set NEO_CONNECTION_URI, NEO_USERNAME, and NEO_PASSWORD in your .env."
        )
    return GraphDatabase.driver(
        NEO_CONNECTION_URI,
        auth=(NEO_USERNAME, NEO_PASSWORD),
    )


def store_in_neo4j(driver, resume_id: str, parsed: ParsedResume) -> None:
    with driver.session() as session:
        session.run(
            """
            MERGE (r:Resume {resume_id: $resume_id})
            SET r.full_name  = $full_name,
                r.email      = $email,
                r.phone      = $phone,
                r.location   = $location,
                r.summary    = $summary,
                r.created_at = datetime()
            MERGE (p:Person {email: $email})
            SET p.full_name = $full_name,
                p.phone     = $phone,
                p.location  = $location
            MERGE (r)-[:BELONGS_TO]->(p)
        """,
            {
                "resume_id": resume_id,
                "full_name": parsed.full_name,
                "email": parsed.email or f"unknown_{resume_id}@placeholder.local",
                "phone": parsed.phone,
                "location": parsed.location,
                "summary": parsed.summary,
            },
        )

        if parsed.skills:
            session.run(
                """
                MATCH (r:Resume {resume_id: $resume_id})
                MATCH (p:Person {email: $email})
                UNWIND $skills AS skill_name
                  MERGE (s:Skill {name: skill_name})
                  MERGE (r)-[:HAS_SKILL]->(s)
                  MERGE (p)-[:HAS_SKILL]->(s)
            """,
                {
                    "resume_id": resume_id,
                    "email": parsed.email or f"unknown_{resume_id}@placeholder.local",
                    "skills": parsed.skills,
                },
            )

        for exp in parsed.work_experience or []:
            exp_id = str(uuid.uuid4())
            session.run(
                """
                MATCH (r:Resume {resume_id: $resume_id})
                MATCH (p:Person {email: $email})
                MERGE (c:Company {name: $company})
                CREATE (e:Experience {
                    exp_id:     $exp_id,
                    role:       $role,
                    company:    $company,
                    start_date: $start_date,
                    end_date:   $end_date,
                    is_current: $is_current,
                    bullets:    $bullets
                })
                MERGE (r)-[:HAS_EXPERIENCE]->(e)
                MERGE (e)-[:AT_COMPANY]->(c)
                MERGE (p)-[:WORKED_AT]->(c)
            """,
                {
                    "resume_id": resume_id,
                    "email": parsed.email or f"unknown_{resume_id}@placeholder.local",
                    "exp_id": exp_id,
                    "role": exp.role,
                    "company": exp.company,
                    "start_date": exp.start_date,
                    "end_date": exp.end_date or "present",
                    "is_current": exp.end_date is None,
                    "bullets": exp.bullets,
                },
            )

        for edu in parsed.education:
            edu_id = str(uuid.uuid4())
            session.run(
                """
                MATCH (r:Resume {resume_id: $resume_id})
                MATCH (p:Person {email: $email})
                MERGE (i:Institution {name: $institution})
                CREATE (e:Education {
                    edu_id:          $edu_id,
                    degree:          $degree,
                    field:           $field,
                    graduation_year: $graduation_year
                })
                MERGE (r)-[:HAS_EDUCATION]->(e)
                MERGE (e)-[:AT_INSTITUTION]->(i)
                MERGE (p)-[:STUDIED_AT]->(i)
            """,
                {
                    "resume_id": resume_id,
                    "email": parsed.email or f"unknown_{resume_id}@placeholder.local",
                    "edu_id": edu_id,
                    "institution": edu.institution,
                    "degree": edu.degree,
                    "field": edu.field or "",
                    "graduation_year": edu.graduation_year or "",
                },
            )

        for cert_name in parsed.certifications or []:
            session.run(
                """
                MATCH (r:Resume {resume_id: $resume_id})
                MERGE (c:Certification {name: $name})
                MERGE (r)-[:HAS_CERTIFICATION]->(c)
            """,
                {"resume_id": resume_id, "name": cert_name},
            )

        for proj in parsed.projects or []:
            proj_id = str(uuid.uuid4())
            session.run(
                """
                MATCH (r:Resume {resume_id: $resume_id})
                MATCH (p:Person {email: $email})
                MERGE (pr:Project {resume_id: $resume_id, name: $name})
                ON CREATE SET pr.project_id = $project_id
                SET pr.role         = $role,
                    pr.description  = $description,
                    pr.technologies = $technologies,
                    pr.bullets      = $bullets,
                    pr.link         = $link
                MERGE (r)-[:HAS_PROJECT]->(pr)
                MERGE (p)-[:BUILT_PROJECT]->(pr)
            """,
                {
                    "resume_id": resume_id,
                    "email": parsed.email or f"unknown_{resume_id}@placeholder.local",
                    "project_id": proj_id,
                    "name": proj.name,
                    "role": proj.role or "",
                    "description": proj.description,
                    "technologies": proj.technologies,
                    "bullets": proj.bullets,
                    "link": proj.link or "",
                },
            )

            if proj.technologies:
                session.run(
                    """
                    MATCH (r:Resume {resume_id: $resume_id})-[:HAS_PROJECT]->(pr:Project {resume_id: $resume_id, name: $project_name})
                    UNWIND $technologies AS tech_name
                      MERGE (s:Skill {name: tech_name})
                      MERGE (pr)-[:USES_SKILL]->(s)
                      MERGE (r)-[:HAS_SKILL]->(s)
                """,
                    {
                        "resume_id": resume_id,
                        "project_name": proj.name,
                        "technologies": proj.technologies,
                    },
                )

    print(f"  Stored graph in Neo4j (resume_id: {resume_id})")
