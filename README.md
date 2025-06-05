# 📚 Book Recommendation Project

This project explores how to answer the question:

**“I like *The Lord of the Rings* — what else should I read?”**

We approach this problem using collaborative filtering methods on the [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset/data) from Kaggle, which includes over one million user-book ratings.

We develop and compare two recommendation strategies:
- **KNN-based similarity**: using book-user interactions and cosine similarity.
- **Lift-based recommendations**: identifying books that are disproportionately liked by specific user segments.

---

## Methods and Structure

- Data cleaning & segmentation (e.g., by author, title keyword, age group)
- Binary read behavior & rating-based modeling
- Lift metric based on relative user engagement
- KNN with cosine similarity
- Popularity penalization (to reduce dominance of bestsellers)

---

## Data Limitations

The dataset contains noise, inconsistencies (e.g., different book title/author variats), and missing metadata (age, rating). We perform only minimal cleaning, focusing on recommendation logic rather than advanced preprocessing.

---


## Files

- **`01_data_exploration.ipynb`** – in-depth exploration of the dataset structure and content
- **`02_LOTR_and_tolkien_fan_analysis.ipynb`** – initial experiments and exploratory ideas focused on Tolkien and LOTR fans
- **`03_age_preprocessing.ipynb`** – estimating missing age groups based on reading preferences (illustrative purpose only)
- **`04_recommendation.ipynb`** – final recommendation models using Lift and KNN, including post-processing and tuning options
- `functions.py` / `metrics.py` – implementation of helper functions and evaluation metrics
