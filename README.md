# Face Recognition

A simple backend service for face recognition. Built using FastAPI, PostgreSQL, [pgvector](https://github.com/pgvector/pgvector), and ONNX for model inference engine.

> **Note:** This project is still under development.

## Quick Start

The projects uses `docker-compose` for development. To start the project, run the following command:

1. Build the docker images:
```bash
docker build -f Dockerfile -t hiseulgi/face-recognition-api:latest .
```

2. Copy the `.env.example` file to `.env` and update the environment variables:
```bash
cp .env.example .env
```

3. Download the pre-trained model from release page and move it to `static` directory:
```bash
bash script/download.sh
```

4. Start the project:
```bash
docker-compose up -d
```

## API Reference

The API documentation is available at `http://localhost:5000`.

### GET All Face Embeddings `(/api/face)`

Get all face embeddings from the database.

#### Request Parameters

None

#### Response

Success Response:

```json
Status: 200 OK

{
  "timestamp": "1719557904.770582",
  "status": "success",
  "detail": "Face embeddings retrieved",
  "data": [
    {
      "id": 2,
      "profile_id": 2,
      "name": "good",
      "recognition_model": "arcface",
      "detection_model": "faceonnx"
    },
    {
      ...
    },
    {
      ...
    }
  ]
}
```

### POST Register Face `(/api/face/register)`

Register a new face embedding to the database from an image file. The image file will be processed to detect the face and extract the face embedding.

#### Request Parameters

- `name`: string (query) - The name of the person.
- `detection_model`: string (query) - The detection model to use. **Only available `faceonnx` for now.**
- `recognition_model`: string (query) - The recognition model to use. **Only available `arcface` for now.**
- `image`: file (form) - The image file to process.

#### Response

Success Response:

```json
Status: 200 OK

{
  "timestamp": "1719557868.213682",
  "status": "success",
  "detail": "Face registered",
  "data": {
    "id": 1,
    "profile_id": 1,
    "name": "good"
  }
}
```

### POST Recognize Face `(/api/face/recognize)`

Recognize a face from an image file. The image file will be processed to detect the face and extract the face embedding. The face embedding will be compared with the face embeddings in the database.

#### Request Parameters

- `detection_model`: string (query) - The detection model to use. **Only available `faceonnx` for now.**
- `recognition_model`: string (query) - The recognition model to use. **Only available `arcface` for now.**
- `image`: file (form) - The image file to process.

#### Response

Success Response:

```json
Status: 200 OK

{
  "timestamp": "1719558134.5086367",
  "status": "success",
  "detail": "Face recognized",
  "data": {
    "id": 1,
    "profile_id": 1,
    "name": "good",
    "distance": 0.3494
  }
}
```

### DELETE Face Embedding `(/api/face/{id})`

Delete a face embedding from the database.

#### Request Parameters

- `id`: integer (path) - The ID of the face embedding.

#### Response

Success Response:

```json
Status: 200 OK

{
  "timestamp": "1719558375.0022182",
  "status": "success",
  "detail": "Face embedding deleted",
}
```

## Acknowledgements

- [pgvector](https://github.com/pgvector/pgvector)
- [FaceONNX](https://github.com/FaceONNX/FaceONNX.Models)
- [ArcFace](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/arcface)

## TODO

- [ ] Add more models for face detection and recognition.
- [ ] Add more recognition methods.
- [ ] Face alignment.
- [ ] Face anti-spoofing.
- [ ] Microservice architecture (separate services for face detection, face recognition, and main API).
