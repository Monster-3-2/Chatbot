"""
Project 3: AI Recommendation Logic
Tech Stack Recommender - DecodeLabs Batch 2026
Uses Content-Based Filtering with TF-IDF + Cosine Similarity
"""

import math
from collections import Counter


# ─────────────────────────────────────────────
#  DATASET  (raw_skills equivalent)
# ─────────────────────────────────────────────
JOB_ROLES = {
    "Data Scientist": [
        "python", "machine learning", "sql", "statistics", "data analysis",
        "tensorflow", "pandas", "numpy", "scikit-learn", "data visualization"
    ],
    "Data Engineer": [
        "python", "sql", "spark", "hadoop", "etl", "data pipelines",
        "aws", "kafka", "airflow", "database"
    ],
    "Machine Learning Engineer": [
        "python", "machine learning", "deep learning", "tensorflow", "pytorch",
        "model deployment", "mlops", "docker", "kubernetes", "algorithms"
    ],
    "Backend Developer": [
        "python", "java", "sql", "apis", "rest", "databases",
        "nodejs", "docker", "microservices", "algorithms"
    ],
    "Frontend Developer": [
        "javascript", "react", "css", "html", "typescript",
        "ui/ux", "web design", "vue", "angular", "responsive design"
    ],
    "Full Stack Developer": [
        "javascript", "python", "react", "nodejs", "sql",
        "html", "css", "apis", "rest", "docker"
    ],
    "DevOps Engineer": [
        "aws", "docker", "kubernetes", "ci/cd", "linux",
        "automation", "terraform", "ansible", "cloud", "git"
    ],
    "Cloud Architect": [
        "aws", "azure", "google cloud", "cloud", "kubernetes",
        "terraform", "networking", "security", "docker", "automation"
    ],
    "Cybersecurity Analyst": [
        "networking", "security", "linux", "ethical hacking", "penetration testing",
        "python", "firewalls", "encryption", "risk assessment", "incident response"
    ],
    "AI Research Scientist": [
        "python", "machine learning", "deep learning", "mathematics",
        "statistics", "pytorch", "tensorflow", "algorithms", "research", "nlp"
    ],
    "NLP Engineer": [
        "python", "nlp", "machine learning", "deep learning", "transformers",
        "bert", "tensorflow", "pytorch", "text processing", "algorithms"
    ],
    "Mobile Developer": [
        "java", "kotlin", "swift", "react native", "flutter",
        "ios", "android", "apis", "ui/ux", "javascript"
    ],
    "Database Administrator": [
        "sql", "database", "mysql", "postgresql", "mongodb",
        "data modeling", "performance tuning", "backup", "security", "oracle"
    ],
    "Systems Administrator": [
        "linux", "windows server", "networking", "automation",
        "scripting", "security", "cloud", "aws", "powershell", "bash"
    ],
    "Blockchain Developer": [
        "solidity", "ethereum", "web3", "javascript", "python",
        "smart contracts", "cryptography", "nodejs", "security", "algorithms"
    ],
}


# ─────────────────────────────────────────────
#  STEP 1 — INGESTION: build vocabulary
# ─────────────────────────────────────────────
def build_vocabulary(roles: dict) -> list:
    vocab = set()
    for skills in roles.values():
        for skill in skills:
            vocab.add(skill.lower())
    return sorted(vocab)


# ─────────────────────────────────────────────
#  TF-IDF helpers
# ─────────────────────────────────────────────
def compute_tf(doc_terms: list) -> dict:
    count = Counter(doc_terms)
    total = len(doc_terms)
    return {term: freq / total for term, freq in count.items()}


def compute_idf(all_docs: list) -> dict:
    N = len(all_docs)
    idf = {}
    all_terms = set(term for doc in all_docs for term in doc)
    for term in all_terms:
        doc_freq = sum(1 for doc in all_docs if term in doc)
        idf[term] = math.log(N / doc_freq)
    return idf


def tfidf_vector(doc_terms: list, vocabulary: list, idf: dict) -> list:
    tf = compute_tf(doc_terms)
    return [tf.get(term, 0) * idf.get(term, 0) for term in vocabulary]


# ─────────────────────────────────────────────
#  STEP 2 — SCORING: cosine similarity
# ─────────────────────────────────────────────
def cosine_similarity(vec_a: list, vec_b: list) -> float:
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a ** 2 for a in vec_a))
    magnitude_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


# ─────────────────────────────────────────────
#  STEPS 3 & 4 — SORTING + FILTERING
# ─────────────────────────────────────────────
def get_recommendations(user_skills: list, top_n: int = 3) -> list:
    user_skills = [s.lower().strip() for s in user_skills]

    all_docs = [skills for skills in JOB_ROLES.values()]
    vocabulary = build_vocabulary(JOB_ROLES)
    idf = compute_idf(all_docs)

    # User profile vector
    user_vector = tfidf_vector(user_skills, vocabulary, idf)

    # Score every job role
    scores = []
    for role, skills in JOB_ROLES.items():
        role_vector = tfidf_vector(skills, vocabulary, idf)
        score = cosine_similarity(user_vector, role_vector)
        scores.append((role, score))

    # Sort descending, return Top-N
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


# ─────────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────────
def display_recommendations(user_skills: list, results: list):
    print("\n" + "=" * 55)
    print("       🤖  TECH STACK RECOMMENDER  —  DecodeLabs")
    print("=" * 55)
    print(f"  Your Skills : {', '.join(user_skills)}")
    print("-" * 55)
    print("  TOP RECOMMENDED CAREER PATHS")
    print("-" * 55)
    medals = ["🥇", "🥈", "🥉"]
    for i, (role, score) in enumerate(results):
        bar = "█" * int(score * 30)
        medal = medals[i] if i < 3 else f"#{i+1}"
        print(f"  {medal}  {role}")
        print(f"      Match: {score:.2%}  {bar}")
        print(f"      Skills needed: {', '.join(JOB_ROLES[role][:5])}...")
        print()
    print("=" * 55)


# ─────────────────────────────────────────────
#  MAIN — get at least 3 inputs from user
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 55)
    print("   🚀  Welcome to the Tech Stack Recommender!")
    print("   Powered by TF-IDF + Cosine Similarity")
    print("=" * 55)
    print("\nAvailable skill examples:")
    print("  python, java, javascript, sql, machine learning,")
    print("  deep learning, aws, docker, kubernetes, react,")
    print("  nodejs, linux, security, nlp, data analysis ...\n")

    # Cold-start bypass — onboarding survey (minimum 3 inputs)
    skills = []
    print("Enter at least 3 skills (one per line). Type 'done' when finished.\n")
    while True:
        prompt = f"  Skill {len(skills) + 1}: "
        skill = input(prompt).strip()
        if skill.lower() == "done":
            if len(skills) < 3:
                print(f"  ⚠️  Please enter at least {3 - len(skills)} more skill(s).")
            else:
                break
        elif skill:
            skills.append(skill)
            if len(skills) >= 3:
                print("  (You can add more or type 'done' to get recommendations)")

    top_n = input("\nHow many recommendations would you like? [default: 3]: ").strip()
    top_n = int(top_n) if top_n.isdigit() else 3

    results = get_recommendations(skills, top_n=top_n)
    display_recommendations(skills, results)


if __name__ == "__main__":
    main()