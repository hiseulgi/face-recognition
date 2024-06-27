CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS profiles (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);


CREATE TABLE IF NOT EXISTS face_embeddings (
    id SERIAL PRIMARY KEY,
    profile_id INTEGER NOT NULL,
    embedding vector(512) NOT NULL,
    detection_model TEXT NOT NULL,
    recognition_model TEXT NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    FOREIGN KEY (profile_id) REFERENCES profiles(id)
);
