# download onnx model
wget https://github.com/hiseulgi/face-recognition/releases/download/1.0.0/arcfaceresnet100-11-int8.onnx
wget https://github.com/hiseulgi/face-recognition/releases/download/1.0.0/face_detector_640.onnx

# download sample images
wget https://github.com/hiseulgi/face-recognition/releases/download/1.0.0/sample_face.zip

# unzip sample images
unzip sample_face.zip

# move to static folder
mv sample_face static/sample_face
mv arcfaceresnet100-11-int8.onnx static/arcfaceresnet100-11-int8.onnx
mv face_detector_640.onnx static/face_detector_640.onnx

# remove zip file
rm sample_face.zip