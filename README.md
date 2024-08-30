
# Tomato Leaves Diseases Detection API

This project implements a YOLO model trained on [Kaggle Tomato Leaves Diseases dataset](https://www.kaggle.com/datasets/kpoviesistphane/tomato-leaf-disease-detection) to detect leaf diseases of tomato plants. Also, the API uses an LLM to generate description of detected diseases.

## Installation

First, create a `.env` file in the root directory and set your Fireworks API key

```text
FIREWORKS_API_KEY=<YOUR FIREWORKS API KEY>
```

### Without Docker
1. python 3.11 is required
2. Install Python dependencies `pip install -r requirements.txt`
3. Download YOLO weights `RUN gdown --fuzzy https://drive.google.com/file/d/1cOClJvYkBxURCa3o4FB83fY-QjknNZAR/view?usp=drive_link`. Or, you can train a model on your own using `train_yolo.ipynb` notebook located in the root directory. **Note: the YOLO weights must be located in the root directory and named `yolo_weights.pt`**
4. Run `python -m src.api.server`

### Docker
1. Run `docker build -t tomato-leaves-api .`
2. Run `docker run -p 5000:5000 tomato-leaves-api`

## API Reference

#### Healtchcheck

Check if the API is running.

```text
  GET /health
```

#### Generate predictions

Predict classes, bboxes and confidence scores of the input image and return LLM response.

```text
POST /api/predict
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `file`      | `image/*` | **Required**. An image to generate predictions for |

