const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const tf = require('@tensorflow/tfjs-node');
const Jimp = require('jimp');

const app = express();
app.use(cors());
app.use(bodyParser.json({ limit: '10mb' })); // to handle large base64 images

let model;

// Load the model at startup
(async () => {
    try {
        model = await tf.loadLayersModel('file://models/model.json');
        console.log('Model loaded successfully');
        app.listen(4000, () => {
            console.log('Express server running on port 4000');
        });
    } catch (error) {
        console.error('Failed to load model:', error);
    }
})();

app.post('/predict', async (req, res) => {
    try {
        const { image } = req.body;
        if (!image) {
            return res.status(400).json({ error: 'No image provided' });
        }

        // image is a base64 data URL (e.g., "data:image/png;base64,...")
        const base64Data = image.replace(/^data:image\/\w+;base64,/, '');
        const buf = Buffer.from(base64Data, 'base64');

        // Use Jimp to read and resize the image
        const img = await Jimp.read(buf);
        // Resize to the model’s input size: 48x48
        img.resize(48, 48);

        // Convert the image to a tensor
        const imageBuffer = tf.node.decodeImage(await img.getBufferAsync(Jimp.MIME_PNG), 3);
        // Normalize if needed: depends on model training
        // For example, if trained on [0,255] images:
        const input = imageBuffer.expandDims(0).toFloat().div(tf.scalar(255));

        // Predict
        const prediction = model.predict(input);
        const scores = prediction.dataSync();

        // Assuming the model outputs a vector of probabilities for each emotion class
        // You’ll need the class labels array you used during training
        const emotionClasses = ['happy', 'sad', 'angry', 'surprised', 'neutral']; // Example classes
        const maxIndex = scores.indexOf(Math.max(...scores));
        const emotion = emotionClasses[maxIndex];

        res.json({ emotion, scores: Array.from(scores) });
    } catch (error) {
        console.error('Error processing image:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

app.listen(4000, () => {
    console.log('Express server running on port 4000');
});
